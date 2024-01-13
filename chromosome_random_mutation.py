from random import randint
import numpy as np 

class ChromosomeRandomMutation:
    def __init__(self, problParam = None):
        self.__problParam = problParam
        self.__repres = []
        self.__baseRepres = problParam.get("baseRepres", None)
        
        if self.__baseRepres is None:
            adj = np.random.uniform(0, 1, (self.__problParam['n'], self.__problParam['n']))
            adj = (adj > (1 - self.__problParam['pc'])).astype(int)
            self.__repres = ((adj + adj.T) >= 1).astype(int)
        else:
            self.__repres = self.__baseRepres

        self.__fitness = 0.0
        self.__metadata = {}

    @property
    def fitness(self):
        return self.__fitness 
    
    @property
    def metadata(self):
        return self.__metadata 
    
    @property
    def repres(self):
        return self.__repres

    @repres.setter
    def repres(self, l = []):
        self.__repres = l 
    
    @fitness.setter 
    def fitness(self, fit = 0.0):
        self.__fitness = fit 
        
    @metadata.setter 
    def metadata(self, metadata = {}):
        self.__metadata = metadata 
    
    def crossover(self, c):
        noDim = self.__problParam['n']
        
        newrepres = np.empty_like(self.__repres)  # Create an empty array of the same shape as __repres
        for idx, (geneM, geneF) in enumerate(zip(self.__repres, c.__repres)):
            cuttingPoint = randint(0, noDim - 1)
            # Using NumPy slicing to create the new gene
            newGene = np.concatenate((geneM[:cuttingPoint], geneF[cuttingPoint:]))
            newrepres[idx] = newGene
        
        self.__problParam["baseRepres"] = newrepres
        offspring = ChromosomeRandomMutation(self.__problParam)

        return offspring
    
    def mutation(self):
        gene = randint(0, len(self.__repres) - 1)
        bit = randint(0, len(self.__repres[0]) - 1)

        self.__repres[gene][bit] = 1 - self.__repres[gene][bit]
        self.__repres[bit][gene] = 1 - self.__repres[bit][gene]
        
    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness
