import random
import numpy as np
import sys
from domain.make_env import make_env
from domain.task_gym import GymTask
from neat_src import *


class WannGymTask(GymTask):
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1): 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """

    GymTask.__init__(self, game, paramOnly, nReps)


# -- 'Weight Agnostic Network' evaluation -------------------------------- -- #
  def setWeights(self, wVec, wVal):
    """Set single shared weight of network
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      wVal    - (float)    - value to assign to all weights
  
    Returns:
      wMat    - (np_array) - weight matrix with single shared weight
                [N X N]
    """
    # Create connection matrix
    wVec[np.isnan(wVec)] = 0
    dim = int(np.sqrt(np.shape(wVec)[0]))    
    cMat = np.reshape(wVec,(dim,dim))
    cMat[cMat!=0] = 1.0

    # Assign value to all weights
    wMat = np.copy(cMat) * wVal 
    return wMat

  def getFitness(self, wVec, aVec, hyp, size, seed=-1, nRep=False,nVals=6,view=False,returnVals=False):
    """Get fitness of a single individual with distribution of weights
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
      hyp     - (dict)     - hyperparameters
        ['alg_wDist']        - weight distribution  [standard;fixed;linspace]
        ['alg_absWCap']      - absolute value of highest weight for linspace
  
    Optional:
      seed    - (int)      - starting random seed for trials
      nReps   - (int)      - number of trials to get average fitness
      nVals   - (int)      - number of weight values to test

  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = hyp['alg_nReps']

    # Set weight values to test WANN with
    if (hyp['alg_wDist'] == "standard") and nVals==6: # Double, constant, and half signal 
      wVals = np.array((-2,-1.0,-0.5,0.5,1.0,2))
    else:
      wVals = np.linspace(-self.absWCap, self.absWCap ,nVals)


    # Get reward from 'reps' rollouts -- test population on same seeds
    reward = np.empty((nRep,nVals))
    stateTable = []
    for iVal in range(nVals):
      stateArray = []
      for iRep in range(nRep):
        wMat = self.setWeights(wVec,wVals[iVal])
        if seed == -1:
          #stateは死ぬまでのフレーム数 x 24
          reward[iRep,iVal], state = self.testInd(wMat, aVec, seed=seed, view=view)
        else:
          reward[iRep,iVal], state = self.testInd(wMat, aVec, seed=seed+iRep, view=view)
        #print('state1', len(state[0])) #24
        for _i in range(len(state)):
          stateArray.append(state[_i])
        #print('aaa', len(stateArray))
      #print(stateArray)
      #stateTableはrepeat x stateArray (デフォルトは4)
      stateTable.append(stateArray)
      #print(len(stateTable))

    #print(stateTable)
    #ミニバッチサイズに圧縮，平均を取る
    batched_state = []
    for _i in range(nVals):
      _b = miniBatchSlave(stateTable[_i], size)
      batched_state.append(_b)
    #print(len(batched_state))

    if returnVals is True:
      return np.mean(reward,axis=0), wVals
    return np.mean(reward,axis=0), batched_state

def miniBatchSlave(Table, size):
  #Table = Table.tolist()
  length = len(Table)
  sum_state = []
  #ミニバッチサイズ分ランダムに入力データを取る
  for _i in range(size):
    r = random.randrange(length)
    sum_state.append(Table[r])
    #print('print5', len(Table[r]))
  #print(len(sum_state))
  #print(sum_state)
  sum_state = np.array(sum_state, dtype=float) 
  length = len(Table[0])
  return sum_state
