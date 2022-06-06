import numpy as np
from network import Network

class Genetic:
    def __init__(self, popsize, elitism, p, K, iter):
        self.popsize = popsize
        self.elitism = elitism
        self.p = p
        self.K = K
        self.iter = iter
        self.population = list()
    
    def fit(self, input, target):
        for i in range(1, self.iter+1):
            helpPopulation = list()
            helpPopulation = self.population[:self.elitism]

            j = len(helpPopulation)

            while len(helpPopulation) != self.popsize:
                weights = list()

                for w1, w2 in zip(self.population[0][0], self.population[j][0]):
                    w = (w1[0] + w2[0]) / 2
                    w = self.mutation(w)
                    b = (w1[1] + w2[1]) / 2
                    b = self.mutation(b)
                    weights.append((w, b))
                j+=1

                net = Network(weights)

                helpPopulation.append((weights, net(input, target)))
                
            self.population = helpPopulation.copy()
            self.population.sort(key = lambda x: x[1])
            
            if i % 2000 == 0:
                print(f"[Train error @{i}]:", self.population[0][1])

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
                #  b1 = np.zeros(shape=dim)
            w2 = np.random.normal(scale=0.01, size=(dim[-1], 1))
            b2 = np.random.normal(scale=0.01, size=(1))
                #  b2 = np.zeros(shape=1)  
            weights.append((w2, b2))      

            net = Network(weights)

            self.population.append((weights, net(input, target)))
            self.population.sort(key = lambda x: x[1])
            
    
    def testPredict(self, input, target):
        bestConf = self.population[0]
        net = Network(bestConf[0])
        print("[Test error]: ", net(input, target))

