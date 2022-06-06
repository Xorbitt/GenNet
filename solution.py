import argparse
import re
import numpy as np
from genetic import Genetic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--nn', required=True)
    parser.add_argument('--popsize', required=True, type=int)
    parser.add_argument('--elitism', required=True, type=int)
    parser.add_argument('--p', required=True, type=float)
    parser.add_argument('--K', required=True, type=float)
    parser.add_argument('--iter', required=True, type=int)

    args = parser.parse_args()
    r = [int(x) for x in re.findall('[0-9]+', args.nn)]
    vars = list()
    input = list() 
    target = list()

    try:
        with open(args.train) as file:
            vars = file.readline().strip('\n').split(',')

            for ln, el in enumerate(file.readlines()):
                elList = [float(x) for x in el.strip('\n').split(',')]
                input.append(elList[:len(elList) - 1])
                target.append(elList[-1])

    except FileNotFoundError:
        print('File not found, check path!')
        raise
    
    input = np.array(input)
    target = np.array(target)

    myGenNet = Genetic(args.popsize, args.elitism, args.p, args.K, args.iter)
    myGenNet.makePopulation(r, input, target, len(vars) - 1)
    myGenNet.fit(input, target)
    
    testTarget = list()
    testInput = list()

    try:
        with open(args.test) as file:
            file.readline().strip('\n').split(',')

            for ln, el in enumerate(file.readlines()):
                elList = [float(x) for x in el.strip('\n').split(',')]
                testInput.append(elList[:len(elList) - 1])
                testTarget.append(elList[-1])

    except FileNotFoundError:
        print('File not found, check path!')
        raise
    
    testTarget = np.array(testTarget)
    testInput = np.array(testInput)

    myGenNet.testPredict(testInput, testTarget)
    