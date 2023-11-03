import numpy as np
import copy
import random
from .ind import *
from .ann import getLayer, getNodeOrder
from utils import listXor
import json
import sys

class WannInd(Ind):
  """Individual class: genes, network, and fitness
  """ 
  def __init__(self, conn, node):
    #rank = self.rank
    #print(node[1, :])
    """Intialize individual with given genes
    Args:
      conn - [5 X nUniqueGenes]
             [0,:] == Innovation Number
             [1,:] == Source
             [2,:] == Destination
             [3,:] == Weight
             [4,:] == Enabled?
      node - [3 X nUniqueGenes]
             [0,:] == Node Id
             [1,:] == Type (1=input, 2=output, 3=hidden, 4=bias)
             [2,:] == Activation function (as int)
  
    Attributes:
      node    - (np_array) - node genes (see args)
      conn    - (np_array) - conn genes (see args)
      nInput  - (int)      - number of inputs
      nOutput - (int)      - number of outputs
      wMat    - (np_array) - weight matrix, one row and column for each node
                [N X N]    - rows: connection from; cols: connection to
      wVec    - (np_array) - wMat as a flattened vector
                [N**2 X 1]    
      aVec    - (np_array) - activation function of each node (as int)
                [N X 1]    
      nConn   - (int)      - number of connections
      fitness - (double)   - fitness averaged over all trials (higher better)
      fitMax  - (double)   - best fitness over all trials (higher better)
      rank    - (int)      - rank in population (lower better)
      birth   - (int)      - generation born
      species - (int)      - ID of species
    """
    Ind.__init__(self,conn,node)
    self.fitMax  = [] # Best fitness over trials

  def createChild(self, p, innov, state, myhyp, gen=0):
    """Create new individual with this individual as a parent

      Args:
        p      - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
        innov  - (np_array) - innovation record
           [5 X nUniqueGenes]
           [0,:] == Innovation Number
           [1,:] == Source
           [2,:] == Destination
           [3,:] == New Node?
           [4,:] == Generation evolved
        gen    - (int)      - (optional) generation (for innovation recording)


    Returns:
        child  - (Ind)      - newly created individual
        innov  - (np_array) - updated innovation record

    """     
    child = WannInd(self.conn, self.node)
    child, innov = child.topoMutate(p,innov,state,gen,myhyp)
    return child, innov

# -- 'Single Weight Network' topological mutation ------------------------ -- #

  def topoMutate(self, p, innov, state, gen, myhyp):
    """Randomly alter topology of individual
    Note: This operator forces precisely ONE topological change 

    Args:
      child    - (Ind) - individual to be mutated
        .conns - (np_array) - connection genes
                  [5 X nUniqueGenes] 
                  [0,:] == Innovation Number (unique Id)
                  [1,:] == Source Node Id
                  [2,:] == Destination Node Id
                  [3,:] == Weight Value
                  [4,:] == Enabled?  
        .nodes - (np_array) - node genes
                  [3 X nUniqueGenes]
                  [0,:] == Node Id
                  [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                  [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                  [5 X nUniqueGenes]
                  [0,:] == Innovation Number
                  [1,:] == Source
                  [2,:] == Destination
                  [3,:] == New Node?
                  [4,:] == Generation evolved

    Returns:
        child   - (Ind)      - newly created individual
        innov   - (np_array) - innovation record

    """

    # Readability
    nConn = np.shape(self.conn)[1]
    connG = np.copy(self.conn)
    nodeG = np.copy(self.node)

    # Choose topological mutation
    topoRoulette = np.array((p['prob_addConn'], p['prob_addNode'], \
                              p['prob_enable'] , p['prob_mutAct']))

    spin = np.random.rand()*np.sum(topoRoulette)
    slot = topoRoulette[0]
    choice = topoRoulette.size
    for i in range(1,topoRoulette.size):
      if spin < slot:
        choice = i
        break
      else:
        slot += topoRoulette[i]

    # Add Connection
    if choice == 1:
      connG, innov = self.mutAddConn(connG, nodeG, innov, gen, p)  

    # Add Node
    elif choice == 2:
      connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p)

    # Enable Connection
    elif choice == 3:
      disabled = np.where(connG[4,:] == 0)[0]
      if len(disabled) > 0:
        enable = np.random.randint(len(disabled))
        connG[4,disabled[enable]] = 1

    # Mutate Activation
    elif choice == 4:
      start = 1+self.nInput + self.nOutput
      end = nodeG.shape[1]           
      if start != end:
        mutNode = np.random.randint(start,end)

        #branchData , act_change_alg = masudaSetup()

        #ここからかきました
        #print(branchData['act_change_alg'])
        if myhyp['act_change_alg'] == 0:     #既存手法
          newActPool = listXor([int(nodeG[2,mutNode])], list(p['ann_actRange']))
          nodeG[2,mutNode] = int(newActPool[np.random.randint(len(newActPool))])

        elif myhyp['act_change_alg'] == 1:   #2条誤差の合計
          #print(nodeG[2,mutNode], end=" ")
          _i = [int(nodeG[2,mutNode])][0] -1

          # print(len(myhyp['activate_ID']))
          # print(myhyp['activate_ID'])
          # print(len(myhyp['activate_table'][_i]))
          # print(myhyp['activate_table'][_i])

          nodeG[2,mutNode] = int(random.choices(myhyp['activate_ID'][0], weights = myhyp['activate_table'][_i])[0])
          #print(nodeG[2,mutNode], end="  ")
        
        elif myhyp['act_change_alg'] == 2:   #入力に対する出力の差
          sys.setrecursionlimit(10000)            #再帰関数のネストの深さを変更
          table1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          table2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          # ランダムなノードの選択
          focusID = nodeG[0, mutNode]
          # ノードの出力を計算
          for _i in range (1, 10) :
            for _j in range(6):
              for _k in range(myhyp['mini_batch_size']):
                table1[_i] += (calculateOutput(connG, nodeG, focusID, _i, _j, state[_j][_k]))
          
          #print(type(nodeG[2,mutNode])) #numpy.float64
          _ = random.choices(table2, weights = table1)[0]
          #print(type(_))
          nodeG[2,mutNode] = _
        
        else:
          nodeG[2,mutNode] = nodeG[2,mutNode]

    child = WannInd(connG, nodeG)
    child.birth = gen

    return child, innov

# -- masuda no jikan ----------------------------------------------------- -- #

def calculateOutput(connG, nodeG, focusID, activateID, weight, state):
  weightList = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
  #o = 0
  w = 0
  a = 0
  _val = 0
  preoutput = 0

  #print("connG")
  #print(connG)
  #print("nodeG")
  #print(nodeG)
  #print("focusID")
  #print(focusID)
  #print("")

  # 目的地がfocusIDのシナプスを抽出
  _datac = []
  for _i in range(connG.shape[1]):
    if connG[2, _i] == focusID:
      row = connG[:, _i]
      _datac.append(row)
  _datac = np.array(_datac)
  #print(_datac)

  # focusIDを使用してfocusノードを探す
  _datan = []
  for _i in range(nodeG.shape[1]):
    if nodeG[0, _i] == focusID:
      row = nodeG[:, _i]
      _datan.append(row)
  _datan = np.array(_datan)
  #print(_datan)

  # 入力層だったら出力は
  if(_datan[0, 1] == 0):
    preoutput = state[focusID]
  
  else:
    # 隠れ層だったら
    # 目的地がfocusのシナプスすべてに対して
    for _i in range(_datac.shape[0]):
      # 配列取得
      #print(_datac)
      _datai = _datac[_i]

      # 出発地のノードを探す
      newfocusID = _datai[1]
      newactivateID = _datai[2]
      _val = calculateOutput(connG, nodeG, newfocusID, newactivateID, weight, state)

      # ノードの出力とシナプス荷重の格納
      preoutput += _val         #ノード出力
      #w += _datai[3]    #重み
      #a += _datai[4]    #有効化してあるか

  preinput = preoutput * weightList[weight]
  output = activate(preinput, activateID)

  return output

def activate(input, ID):
  """
  case 1  -- Linear
  case 2  -- Unsigned Step Function
  case 3  -- Sin
  case 4  -- Gausian with mean 0 and sigma 1
  case 5  -- Hyperbolic Tangent [tanh] (signed)
  case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  case 7  -- Inverse
  case 8  -- Absolute Value
  case 9  -- Relu
  case 10 -- Cosine
  case 11 -- Squared
  """
  x = input

  if ID == 1:   # Linear
    value = x

  elif ID == 2:   # Unsigned Step Function
    value = 1.0*(x>0.0)
    #value = (np.tanh(50*x/2.0) + 1.0)/2.0

  elif ID == 3: # Sin
    value = np.sin(np.pi*x) 

  elif ID == 4: # Gaussian with mean 0 and sigma 1
    value = np.exp(-np.multiply(x, x) / 2.0)

  elif ID == 5: # Hyperbolic Tangent (signed)
    value = np.tanh(x)     

  elif ID == 6: # Sigmoid (unsigned)
    value = (np.tanh(x/2.0) + 1.0)/2.0

  elif ID == 7: # Inverse
    value = -x

  elif ID == 8: # Absolute Value
    value = abs(x)   
    
  elif ID == 9: # Relu
    value = np.maximum(0, x)   

  elif ID == 10: # Cosine
    value = np.cos(np.pi*x)

  elif ID == 11: # Squared
    value = x**2
    
  else:
    value = x

  return value