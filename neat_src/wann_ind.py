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
    # print(node[1, :])
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

  def createChild(self, p, innov, gen=0):
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
    child, innov = child.topoMutate(p,innov,gen)
    return child, innov

# -- 'Single Weight Network' topological mutation ------------------------ -- #

  def topoMutate(self, p, innov,gen):
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
        branchData , act_change_alg = masudaSetup()

        #ここからかきました
        #print(branchData['act_change_alg'])
        if branchData['act_change_alg'] == 0:     #既存手法
          newActPool = listXor([int(nodeG[2,mutNode])], list(p['ann_actRange']))
          nodeG[2,mutNode] = int(newActPool[np.random.randint(len(newActPool))])

        elif branchData['act_change_alg'] == 1:   #-3から3の積分
          #print(nodeG[2,mutNode], end=" ")
          _i = [int(nodeG[2,mutNode])][0] -1
          nodeG[2,mutNode] = int(random.choices(act_change_alg[0], weights = act_change_alg[1][_i])[0])
          #print(nodeG[2,mutNode], end="  ")
        
        elif branchData['act_change_alg'] == 2:   #-2から2の積分
          _i = [int(nodeG[2,mutNode])][0] -1
          nodeG[2,mutNode] = int(random.choices(act_change_alg[0], weights = act_change_alg[2][_i])[0])
        
        elif branchData['act_change_alg'] == 3:   #入力に対する出力の差
          sys.setrecursionlimit(10000)            #再帰関数のネストの深さを変更
          table1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          table2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          # ランダムなノードの選択
          focusID = nodeG[0, mutNode]
          # ノードの出力を計算
          for _i in range (10) :
            table1[_i] = calculateOutput(connG, nodeG, focusID)
          
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
def masudaSetup():
  with open('p/masuda/branch.json', 'r') as json_file:
    data = json.load(json_file)

  table = []
  table.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  table.append([
    [0.0   , 0.1294, 0.0904, 0.0785, 0.2763, 0.1142, 0.0216, 0.0431, 0.1726, 0.074 ],
    [0.019 , 0.0   , 0.0484, 0.1006, 0.0956, 0.5955, 0.0076, 0.019 , 0.0762, 0.0381],
    [0.0472, 0.1718, 0.0   , 0.1702, 0.1437, 0.1799, 0.0327, 0.0387, 0.0805, 0.1353],
    [0.0247, 0.2153, 0.1026, 0.0   , 0.0846, 0.3579, 0.0247, 0.0309, 0.0556, 0.1035],
    [0.1049, 0.2466, 0.1044, 0.102 , 0.0   , 0.1935, 0.0154, 0.0268, 0.1224, 0.0841],
    [0.016 , 0.5685, 0.0484, 0.1596, 0.0716, 0.0   , 0.0082, 0.0197, 0.0651, 0.0428],
    [0.0464, 0.1114, 0.1346, 0.1689, 0.087 , 0.1256, 0.0   , 0.0928, 0.0742, 0.1591],
    [0.0512, 0.1537, 0.0878, 0.1166, 0.0838, 0.1662, 0.0512, 0.0   , 0.2049, 0.0845],
    [0.0807, 0.242 , 0.072 , 0.0826, 0.1508, 0.2167, 0.0161, 0.0807, 0.0   , 0.0585],
    [0.0431, 0.1508, 0.1508, 0.1913, 0.1291, 0.1775, 0.0431, 0.0415, 0.0729, 0.0   ],
  ]) #分解能0.001，-3から3の積分
  table.append([
    [0.0   , 0.1165, 0.0393, 0.0547, 0.4467, 0.0898, 0.0182, 0.0364, 0.1456, 0.0529],  
    [0.044 , 0.0   , 0.0367, 0.1069, 0.1032, 0.3956, 0.0129, 0.044 , 0.2201, 0.0367], 
    [0.055 , 0.1359, 0.0   , 0.1444, 0.1041, 0.1456, 0.1135, 0.0741, 0.0915, 0.1359],  
    [0.0345, 0.1785, 0.0651, 0.0   , 0.0639, 0.4015, 0.0345, 0.0673, 0.0906, 0.0642],  
    [0.2904, 0.1776, 0.0484, 0.0658, 0.0   , 0.1319, 0.0181, 0.0341, 0.1717, 0.062 ],  
    [0.031 , 0.3621, 0.036 , 0.22  , 0.0701, 0.0   , 0.0152, 0.0522, 0.1719, 0.0415],  
    [0.0481, 0.0905, 0.2142, 0.1445, 0.0736, 0.1163, 0.0   , 0.0961, 0.0769, 0.1398],  
    [0.0485, 0.1551, 0.0705, 0.1421, 0.0698, 0.201 , 0.0485, 0.0   , 0.1939, 0.0706],  
    [0.0744, 0.2978, 0.0334, 0.0734, 0.1349, 0.2542, 0.0149, 0.0744, 0.0   , 0.0425],  
    [0.0703, 0.1289, 0.1289, 0.1351, 0.1266, 0.1592, 0.0703, 0.0703, 0.1105, 0.0   ]
  ]) #分解能0.001，-2から2の積分

  return data, table

def calculateOutput(connG, nodeG, focusID):
  _listo = []
  _listw = []
  _lista = []
  _val = 0

  # focusID を使用してfocusが目的地のシナプスを探す
  _indexc = np.where(connG[2, :] == focusID)
  _datac = connG[:, _indexc]

  # focusIDを使用してfocusを探す
  _indexn = np.where(nodeG[0, :] == focusID)
  _datan = nodeG[:, _indexn]

  # 入力層だったら出力は
  if(_datan[1, 0] == 0):
    output = 1 # 本来ここには入力ノードの出力が出ていないといけない
    return output

  # 隠れ層だったら
  # 目的地がfocusのシナプスすべてに対して
  for _i in range(len(_indexc[0])):
    # 配列取得
    print(_datac)
    _datai = _datac[:, _i]

    # 出発地のノードを探す
    newfocusID = _datac[1]
    _val = calculateOutput(connG, nodeG, newfocusID)

    # ノードの出力とシナプス荷重の格納
    _listo.append(_val)         #ノード出力
    _listw.append(_datai[3])    #重み
    _lista.append(_datai[4])    #有効化してあるか

  output = 0
  for _i in range(len(_listo)):
    if _lista[_i] == 1:
      output = output + _listo[_i] * _listw[_i]

  return output