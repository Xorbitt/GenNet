import numpy as np
from genNet.network import Network

class Genetic:
    def __init__(self, popsize, elitism, p, K, iter):
        self.popsize = popsize
        self.elitism = elitism
        self.p = p
        self.K = K
        self.iter = iter
        self.population = list()
    
    def fit(self, input, target):
        """
        Backward pass implemented using genetic algorithm
        """
        for i in range(1, self.iter+1):
            helpPopulation = list()
            helpPopulation = self.population[:self.elitism]

            j = len(helpPopulation)

            while len(helpPopulation) != self.popsize:
                weights = list()

                w1, w2 = self.selection()
                
                for w11, w22 in zip(w1,w2):
                    w = (w11[0] + w22[0]) / 2
                    w = self.mutation(w)
                    b = (w11[1] + w22[1]) / 2
                    b = self.mutation(b)
                    weights.append((w, b))

                j+=1

                net = Network(weights)

                helpPopulation.append((weights, net(input, target)[0]))
                
            self.population = helpPopulation.copy()
            self.population.sort(key = lambda x: x[1])
            
            if i % 2000 == 0:
                print(f"[Train error @{i}]:", self.population[0][1])
    
    def selection(self):
        j = 0.0
        l = list() 
        hold = 0.0

        for i in self.population:
            j += i[1]
        
        for i in self.population:
            hold += (i[1] - self.population[-1][1]) / (self.population[0][1] - self.population[-1][1]) 
            l.append(hold)

        w1 = self.getVal(l)
        w2 = self.getVal(l)

        return self.population[w1][0], self.population[w2][0]

    def getVal(self, l):
        r = np.random.rand()

        for n, i in enumerate(l): 
            if i >= r:
                return n

    def mutation(self, el):
        if np.random.rand() >= self.p:
            mutation = np.random.normal(scale=self.K, size=el.shape)
            el += mutation
        
        return el
    
    def makePopulation(self, dim, input, target, varLen):
        while len(self.population) != self.popsize:
            weights = list()
            varHelp = varLen
            for i in dim:
                w1 = np.random.normal(scale=0.01, size=(varHelp, i))
                b1 = np.random.normal(scale=0.01, size=(i))
                weights.append((w1, b1))
                varHelp = i
            w2 = np.random.normal(scale=0.01, size=(dim[-1], 1))
            b2 = np.random.normal(scale=0.01, size=(1))

            weights.append((w2, b2))      

            net = Network(weights)

            self.population.append((weights, net(input, target)[0]))
            self.population.sort(key = lambda x: x[1])
            
    
    def testPredict(self, input, target):
        bestConf = self.population[0]
        net = Network(bestConf[0])
        error, prediction = net(input, target)
        print("[Test error]: ", error)
        return error, prediction
        
        
