import numpy as np
import copy
import random
from .ind import *
from .ann import getLayer, getNodeOrder
from utils import listXor
import json
import sys

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

def main():
  table1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  table2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  # ランダムなノードの選択
  focusID = nodeG[0, mutNode]
  # ノードの出力を計算
  for _i in range (1, 10) :
    for _j in range(6):
      for _k in range(myhyp['mini_batch_size']):
        table1[_i] += (calculateOutput(connG, nodeG, focusID, _i, _j, state[_j][_k]))
    table1[_i] += epsilon
  
  #print(type(nodeG[2,mutNode])) #numpy.float64
  _ = random.choices(table2, weights = table1)[0]
  #print(type(_))
  nodeG[2,mutNode] = _

else:
  nodeG[2,mutNode] = nodeG[2,mutNode]