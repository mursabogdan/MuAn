from random import randint
from chromosome_random_mutation import ChromosomeRandomMutation as Chromosome
import numpy as np
import random

class GA:
    def __init__(self, param = None, problParam = None):
        self.__param = param
        self.__problParam = problParam
        self.__population = []
        
        if self.__problParam.get("baseRepres", None) is not None:
            numpy_repres = np.array(self.__problParam["baseRepres"])
            
            print("Evolving chromosomes from a base with sizes N: {} M: {}".format(numpy_repres.shape[0], numpy_repres.shape[1]))
            
    @property
    def population(self):
        return self.__population
    
    def initialisation(self):
        for _ in range(0, self.__param['p']):
            c = Chromosome(self.__problParam)
            self.__population.append(c)

    def evaluation(self):
        for c in self.__population:
            result = self.__problParam['function'](c.repres)

            c.fitness = result[0]
            c.metadata = result[1]
            
            
    def bestChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness < best.fitness):
                best = c
        return best
        
    def worstChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness > best.fitness):
                best = c
        return best

    def selection(self):
        pos1 = randint(0, self.__param['p'] - 1)
        pos2 = randint(0, self.__param['p'] - 1)
        if (self.__population[pos1].fitness < self.__population[pos2].fitness):
            return pos1
        else:
            return pos2 
        
    
    def oneGeneration(self):
        newPop = []
        for _ in range(self.__param['p']):
            p1 = self.__population[self.selection()]
            newP = Chromosome(self.__problParam)
            newP.repres = p1.repres

            prob = random.uniform(0, 1)
            if prob > 1 - self.__param['pm']:
                newP.mutation()

            newPop.append(newP)
        self.__population = newPop
        self.evaluation()

    def oneGenerationElitism(self):
        newPop = [self.bestChromosome()]
        for _ in range(self.__param['p'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            
            off = p1.crossover(p2)
            
            prob = random.uniform(0, 1)
            if prob > 1 - self.__param['pm']:
                off.mutation()

            newPop.append(off)
    
        self.__population = newPop
        self.evaluation()
        
    def oneGenerationSteadyState(self):
        for _ in range(self.__param['p']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            off.fitness = self.__problParam['function'](off.repres)
            worst = self.worstChromosome()
            if (off.fitness < worst.fitness):
                worst = off
