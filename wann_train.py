import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np
np.set_printoptions(precision=2, linewidth=160) 
import json

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# prettyNeat
from neat_src import * # NEAT and WANNs
from domain import *   # Task environments

# -- Run NEAT ------------------------------------------------------------ -- #
def master(): 
  """Main NEAT optimization script
  """
  global fileName, hyp, myHyp
  data = WannDataGatherer(fileName, hyp)
  alg  = Wann(hyp, myHyp)
  state = []

  epsilon = myHyp['epsilon']

  for gen in range(hyp['maxGen']):        
    pop = alg.ask(state, epsilon)            # 新しく進化生成した個体集団の取得
    reward, state = batchMpiEval(pop)  # 生成した個体の評価
    alg.tell(reward)           # 評価による個体の評価    

    data = gatherData(data,alg,gen,hyp) #データの更新
    print(gen, ' ', data.display())

    epsilon *= myHyp['ratio']

  # Clean up and data gathering at run end
  data = gatherData(data,alg,gen,hyp,savePop=True)
  data.save()
  data.savePop(alg.pop,fileName) # Save population as 2D numpy arrays
  stopAllWorkers()

def gatherData(data,alg,gen,hyp,savePop=False):
  """Collects run data, saves it to disk, and exports pickled population

  Args:
    data       - (DataGatherer)  - collected run data
    alg        - (Wann)          - neat algorithm container
      .pop     - [Ind]           - list of individuals in population    
      .species - (Species)       - current species
    gen        - (ind)           - current generation
    hyp        - (dict)          - algorithm hyperparameters
    savePop    - (bool)          - save current population to disk?

  Return:
    data - (DataGatherer) - updated run data
  """
  data.gatherData(alg.pop, alg.species)
  if (gen%hyp['save_mod']) == 0:
    data = checkBest(data)
    data.save(gen)

  if savePop is True: # Get a sample pop to play with in notebooks    
    global fileName
    pref = 'log/' + fileName
    import pickle
    with open(pref+'_pop.obj', 'wb') as fp:
      pickle.dump(alg.pop,fp)

  return data

def checkBest(data):
  """Checks better performing individual if it performs over many trials.
  Test a new 'best' individual with many different seeds to see if it really
  outperforms the current best.

  Args:
    data - (DataGatherer) - collected run data

  Return:
    data - (DataGatherer) - collected run data with best individual updated


  * This is a bit hacky, but is only for data gathering, and not optimization
  """
  global filename, hyp
  if data.newBest is True:
    bestReps = max(hyp['bestReps'], (nWorker-1))
    rep = np.tile(data.best[-1], bestReps)
    # ここが適応度と入力データを持ってきている
    fitVector, stateVector = batchMpiEval(rep, sameSeedForEachIndividual=False)
    trueFit = np.mean(fitVector)
    if trueFit > data.best[-2].fitness:  # Actually better!      
      data.best[-1].fitness = trueFit
      data.fit_top[-1]      = trueFit
      data.bestFitVec = fitVector
    else:                                # Just lucky!
      prev = hyp['save_mod']
      data.best[-prev:]    = data.best[-prev]
      data.fit_top[-prev:] = data.fit_top[-prev]
      data.newBest = False
  return data


# -- Parallelization ----------------------------------------------------- -- #
def batchMpiEval(pop, sameSeedForEachIndividual=True,):
  """Sends population to workers for evaluation one batch at a time.

  Args:
    pop - [Ind] - list of individuals
      .wMat - (np_array) - weight matrix of network
              [N X N] 
      .aVec - (np_array) - activation function of each node
              [N X 1]

  Return:
    reward  - (np_array) - fitness value of each individual
              [N X 1]

  Todo:
    * Asynchronous evaluation instead of batches
  """
  global nWorker, hyp, myHyp
  nSlave = nWorker-1
  nJobs = len(pop)
  nBatch= math.ceil(nJobs/nSlave) # First worker is master

  # Set same seed for each individual
  if sameSeedForEachIndividual is False:
    seed = np.random.randint(1000, size=nJobs)
  else:
    seed = np.random.randint(1000)

  reward = np.empty( (nJobs,hyp['alg_nVals']), dtype=np.float64)
  state = []

  i = 0 # Index of fitness we are filling
  for iBatch in range(nBatch): # Send one batch of individuals
    for iWork in range(nSlave): # (one to each worker if there)
      if i < nJobs:
        wVec   = pop[i].wMat.flatten()
        n_wVec = np.shape(wVec)[0]
        aVec   = pop[i].aVec.flatten()
        n_aVec = np.shape(aVec)[0]

        comm.send(n_wVec, dest=(iWork)+1, tag=1)
        comm.Send(  wVec, dest=(iWork)+1, tag=2)
        comm.send(n_aVec, dest=(iWork)+1, tag=3)
        comm.Send(  aVec, dest=(iWork)+1, tag=4)
        if sameSeedForEachIndividual is False:
          comm.send(seed.item(i), dest=(iWork)+1, tag=5)
        else:
          comm.send(  seed, dest=(iWork)+1, tag=5)        
      else: # message size of 0 is signal to shutdown workers
        n_wVec = 0
        comm.send(n_wVec,  dest=(iWork)+1)
      i = i+1 
  
    # Get fitness values back for that batch
    i -= nSlave
    for iWork in range(nSlave):
      if i < nJobs:
        workResult = np.empty(hyp['alg_nVals'], dtype='d')
        statelen = np.empty(1, dtype = np.int32)   
        comm.Recv(workResult, source=(iWork)+1, tag=0)
        comm.Recv(statelen, source=(iWork)+1, tag=1)
        stateTable = np.empty((statelen[0], myHyp['mini_batch_size'], myHyp['inputnode_size']), dtype = np.float64)
        for _i in range(statelen[0]):
          _t = 2 + _i
          comm.Recv(stateTable[_i], source=(iWork)+1, tag=_t)
          #print(_t)
        #stateTable = stateTable.tolist()

        #print(stateTable)
        #print('aaa', workResult)

        state.append(stateTable)
        reward[i,:] = workResult
      i+=1
  
  #print(state[0])
  #print(reward[0])
  # rewardの形は個体数 x -2,-1.0,-0.5,0.5,1.0,2 の 6個の共有重みの適応度
  return reward, state

def slave():
  """Evaluation process: evaluates networks sent from master process. 

  PseudoArgs (recieved from master):
    wVec   - (np_array) - weight matrix as a flattened vector
             [1 X N**2]
    n_wVec - (int)      - length of weight vector (N**2)
    aVec   - (np_array) - activation function of each node 
             [1 X N]    - stored as ints, see applyAct in ann.py
    n_aVec - (int)      - length of activation vector (N)
    seed   - (int)      - random seed (for consistency across workers)

  PseudoReturn (sent to master):
    result - (float)    - fitness value of network
  """
  global hyp, myHyp
  mini_batch_size = myHyp['mini_batch_size']
  task = WannGymTask(games[hyp['task']], nReps=hyp['alg_nReps'])

  # Evaluate any weight vectors sent this way
  while True:
    n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
    if n_wVec > 0:
      wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
      comm.Recv(wVec, source=0,  tag=2) # recieve weights

      n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
      aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
      comm.Recv(aVec, source=0,  tag=4) # recieve it
      seed = comm.recv(source=0, tag=5) # random seed as int

      result, state = task.getFitness(wVec,aVec,hyp, mini_batch_size, seed=seed) # process it
      #print(state)
      #state追加に伴いtag追加しました
      comm.Send(result, dest=0, tag=0)            # send it back
      #print(len(state))
      comm.Send(np.int32(len(state)), dest=0, tag=1)
      for _i in range(len(state)):
        _t = 2 + _i
        comm.Send(state[_i], dest=0, tag=_t)

    if n_wVec < 0: # End signal recieved
      print('Worker # ', rank, ' shutting down.')
      break

def stopAllWorkers():
  """Sends signal to all workers to shutdown.
  """
  global nWorker
  nSlave = nWorker-1
  print('stopping workers')
  for iWork in range(nSlave):
    comm.send(-1, dest=(iWork)+1, tag=1)

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    #print('assigning the rank and nworkers', nWorker, rank)
    return "child"

def myDistance(resolution, calc_range, i, j):
  num = 10

  func = [['Linear',               'x'],
          ['UnsignedStepFunction', '1.0 * (x > 0.0)'],
          ['Sin',                  'np.sin(np.pi * x)'],
          ['Gausian',              'np.exp(-np.multiply(x, x) / 2.0)'],
          ['HyperbolicTangent',    'np.tanh(x)'],
          ['SigmoidUnsigned',      '(np.tanh(x / 2.0) + 1.0) / 2.0'],
          ['Inverse',              '-x'],
          ['AbsoluteValue',        'abs(x)'],
          ['Relu',                 'np.maximum(0, x)'],
          ['Cosine',               'np.cos(np.pi * x)'],
          ['Squared',              'x**2']]

  # 距離の計算
  x = -calc_range
  sum = 0.0
  while 1:
      _a = eval(func[i][1])
      _b = eval(func[j][1])
      _diff = (_a - _b)**2
      sum = sum + _diff

      x = x + resolution
      if x > calc_range:
          break

  # イプシロンの追加
  # sum += epsilon

  #逆数を取る
  if(sum != 0):
    sum = 1 / sum

  #自身の値を0にする
  if(i == j):
    sum = 0

  return sum

def loadMyHyp(MyHypPass):
  with open(MyHypPass, 'r') as json_file:
    MyHyp = json.load(json_file)
  
  resolution = MyHyp['resolution']
  calc_range = MyHyp['calc_range']
  #epsilon = MyHyp['epsilon']

  _ = []
  _.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  MyHyp['activate_ID'] = _

  _ = [[0] * 10 for i in range(10)]

  for _i in range(len(_)):
    for _j in range(len(_[0])):
        _d = myDistance(resolution, calc_range, _i, _j)
        _[_i][_j] = _d
  
  for _i in range(len(_)):
    _s = sum(_[_i])
    for _j in range(len(_[0])):
      _[_i][_j] = _[_i][_j] / _s
  
  MyHyp['activate_table'] = _
  
  return MyHyp

# -- Input Parsing ------------------------------------------------------- -- #

def main(argv):
  """Handles command line input, launches optimization or evaluation script
  depending on MPI rank.
  """

  global fileName, hyp# Used by both master and slave processes
  fileName    = args.outPrefix
  hyp_default = args.default
  hyp_adjust  = args.hyperparam

  hyp = loadHyp(pFileName=hyp_default)
  updateHyp(hyp,hyp_adjust)

  # 自分のハイパーパラメータ
  print('Loading myHyp...')
  global myHyp
  myHyp = loadMyHyp(args.masuda)
  print('Ready. core' + str(rank))
  # print(myHyp['activate_table'])

  # Launch main thread and workers
  if (rank == 0):
    master()
  else:
    slave()

if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Evolve WANNs'))
  
  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='p/default_wann.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default='p/laptop_swing.json')

  parser.add_argument('-o', '--outPrefix', type=str,\
   help='file name for result output', default='test')
  
  parser.add_argument('-n', '--num_worker', type=int,\
   help='number of cores to use', default=11)
  
  parser.add_argument('-m', '--masuda', type=str,\
   help='卒業研究のために追加', default='p/masuda/branch.json')

  args = parser.parse_args()


  # Use MPI if parallel
  if "parent" == mpi_fork(args.num_worker+1): os._exit(0)

  main(args)                              
  




