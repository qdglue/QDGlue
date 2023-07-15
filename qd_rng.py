# if you initialize me with:
# N dimension of parameter space
# D dimension of descriptor space
# F dimension of fitness objective space
#
# given a vector with N items, I will return vectors with D and F items

import numpy as np

from qd_task import QDTask


class QDRNG:
    # our expectation
    # evaLFn is a class that contains
    '''
    N 
    D 
    F
    evalFn
    Binary?
    Archive is in EvalInstance
    '''

    def __init__(self, evalInstance):
        self.evalInstance = evalInstance
        return

    def emit(self):
        parameterSpaceDim = self.evalInstance.N
        #descriptorSpaceDim =  self.evalInstance.D
        #fitnessSpaceDim = self.evalInstance.F
        isBinary = self.B

        parameters = np.random.random(parameterSpace)
        #note To Future Self: make elegant
        if isBinary:
            parameters[parameters < 0.5] = 0
            parameters[parameters >= 0.5] = 1

        return parameters

    def add(self, parameters, measures, fitness):
        '''
        parameters is the parameters we just evaluted
        measures is the objective mesures
        fitness the fitness measure 
        '''
        return None


def main():
    print("hello")


if __name__ == "__main__":

    main()
