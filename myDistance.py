import numpy
import math
import tabulate
import random
    
def inputData():
    print('input 分解能', end = ' : ')
    r = input()
    print('input 有効範囲', end = ' : ')
    e = input()
    print('input 類似方法(0:積分 1:最小 2:最大 3:シナプス入力に対する出力の差)', end = ' : ')
    m = input()
    print('')
    return float(r), float(e), float(m)

def createRandom(effect):
    print("input : ", end = "")
    inputList = []
    for i in range(5):
        inputList.append(random.gauss(0, effect))
        print(inputList[i], end = ", ")
    print("")
    
    return inputList

def similarity(func, i, j, resolution, effect, method, inputList):
    if method == 0: #有効区間内の積分
        x = -effect
        sum = 0.0
        while 1:
            _a = eval(func[i][1])
            _b = eval(func[j][1])
            _diff = (_a - _b)**2
            sum = sum + _diff

            x = x + resolution
            if x > effect:
                break
        return sum

    if method == 1: #有効区間内の最小の差
        x = -effect
        diff = 10.0
        while 1:
            _a = eval(func[i][1])
            _b = eval(func[j][1])
            _diff = (_a - _b)**2
            if _diff < diff:
                diff = _diff
            
            x = x + resolution
            if x > effect:
                break
        return diff
    
    if method == 2: #有効区間内の最大の差
        x = -effect
        diff = 0.01
        while 1:
            _a = eval(func[i][1])
            _b = eval(func[j][1])
            _diff = (_a - _b)**2
            if _diff > diff:
                diff = _diff
            
            x = x + resolution
            if x > effect:
                break
        return diff

    if method == 3: #シナプス荷重を考慮した入力に対する出力の差
        diff = 0.0
        for k in range(len(inputList)):
            x = inputList[k]
            _a = eval(func[i][1])
            _b = eval(func[j][1])
            _diff = (_a - _b)**2
            diff = diff + _diff

        return diff
    
    else:
        return 1.0

def normalize(NUM, mylist):
    _nor = [0]*NUM
    _max = mylist[0][0]
    for i in range(NUM):
        for j in range(NUM):
            if mylist[i][j] > _max:
                _max = mylist[i][j]

    for i in range(NUM):
        for j in range(NUM):
            mylist[i][j] = mylist[i][j] / _max
    
    for i in range(NUM):
        for j in range(NUM):
            if mylist[i][j] != 0.0:
                mylist[i][j] = 1.0 / mylist[i][j]
            _nor[i] = _nor[i] + mylist[i][j]
        for j in range(NUM):
            mylist[i][j] = mylist[i][j] / _nor[i]

def output(NUM, mylist, func):
    for i in range(NUM):
        print (i,func[i][0].ljust(20),func[i][1])
    print('')

    for i in range(NUM):
        for j in range(NUM):
            _out = '{:.4}'.format(mylist[i][j])
            _out = round(mylist[i][j], 4)
            _out = str(_out).ljust(6)
            print(_out, end = '  ')
        print('')

def main():
    NUM = 10

    mylist = [[0 for _i in range(NUM)] for _j in range(NUM)]
    func = [['Linear',               'x'],
            ['UnsignedStepFunction', '1.0 * (x > 0.0)'],
            ['Sin',                  'numpy.sin(numpy.pi * x)'],
            ['Gausian',              'numpy.exp(-numpy.multiply(x, x) / 2.0)'],
            ['HyperbolicTangent',    'numpy.tanh(x)'],
            ['SigmoidUnsigned',      '(numpy.tanh(x / 2.0) + 1.0) / 2.0'],
            ['Inverse',              '-x'],
            ['AbsoluteValue',        'abs(x)'],
            ['Relu',                 'numpy.maximum(0, x)'],
            ['Cosine',               'numpy.cos(numpy.pi * x)'],
            ['Squared',              'x**2']]

    resolution, effect, method = inputData()

    inputList = []
    if method == 3:
        inputList = createRandom(effect)

    for i in range(0, NUM):
        for j in range(i, NUM):
            mylist[i][j] = similarity(func, i, j, resolution, effect, method, inputList)
            mylist[j][i] = mylist[i][j]
    
    normalize(NUM, mylist)
    output(NUM, mylist, func)

def test(resolution, calc_range):
    return 0

if __name__ == '__main__':
    main()