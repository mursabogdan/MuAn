import networkx as nx
import numpy as np
import math
from ga import GA


import numpy as np
import math

from time import time



def compute_euclidian_distance(x, y):
    p1 = [SCORE_THRESHOLD, SCORE_THRESHOLD]
    p2 = [x, y]

    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def fcEval(x):
    G = nx.from_numpy_array(x, create_using=nx.Graph())

    ccl_by_node = nx.clustering(G)
    clustering_coefficient = sum(ccl_by_node.values())/len(ccl_by_node.values())

    rho = nx.degree_assortativity_coefficient(G)

    return compute_euclidian_distance(rho, clustering_coefficient), {"rho": rho, "clustering_coefficient": clustering_coefficient}

global ga
def start_ga(noDim=2, threshold=0.2, fitnesses=None, epsilon=0.01, pc=0.1, pm=0.2, popSize=100):

    # initialise de GA parameters
    # population size (p)
    # number of generation for evolution (Gen_no)
    gaParam = {'p' : popSize, 'gen_no' : 10000, 'pm' : pm, 'fs': threshold}

    # problem parameters:
    # pm - mutation probability
    # noDim - the number of nodes (n)
    problParam = {'function' : fcEval, 'n' : noDim, 'pc' : pc}

    print(gaParam)
    print(problParam)

    ga = GA(gaParam, problParam)
    ga.initialisation()
    ga.evaluation()
    start = time()
    for g in range(gaParam['gen_no']):
        #logic alg
#         ga.oneGeneration()
        ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()

        bestChromo = ga.bestChromosome()

        fitnesses['gen'].append(g)
        fitnesses['rho'].append(bestChromo.metadata['rho'])
        fitnesses['clustering_coefficient'].append(bestChromo.metadata['clustering_coefficient'])
        fitnesses['fitness_score_diff'].append(bestChromo.fitness)

        if bestChromo.fitness <= epsilon: # this is epsilon - premature stop condition
            break

        if g % 50 == 0:
            meta = bestChromo.metadata
            end = time()
            print(f"{g}/{gaParam['gen_no']} ({round(g/gaParam['gen_no'] * 100, 2)}%) ---- " + 'Best solution in generation ' + str(g) + ' f(x) = ' + str(bestChromo.fitness) + ' vs ' + str(epsilon) + ' (metadata ' + str(meta) + ' ) ' + str(end - start), flush=True)
            start = time()

    print('Best solution in generation ' + str(g) + ' f(x) = ' + str(threshold - bestChromo.fitness) + ' vs ' + str(threshold), flush=True)


    return nx.from_numpy_array(bestChromo.repres)


def compute_general_metrics(i, G, res):
    ccl_by_node = nx.clustering(G)
    clustering_coefficient = sum(ccl_by_node.values())/len(ccl_by_node.values())

    degree_centrality_by_node = nx.degree_centrality(G)
    degree_centrality = sum(degree_centrality_by_node.values())/len(degree_centrality_by_node.values())

    assort_degree = nx.degree_assortativity_coefficient(G)
    density = nx.density(G)

    if res is not None:
        res['degree_centrality'].append(degree_centrality)
        res['clustering_coefficient'].append(clustering_coefficient)
        res['assort_degree'].append(assort_degree)
        res['density'].append(density)

    print("Module {}.\n\tDegree centrality: {}\n\tCCL: {}\n\tRho:{}\n\tDensity: {}".format(i, degree_centrality, clustering_coefficient, assort_degree, density), flush=True)

    return assort_degree, clustering_coefficient, density, degree_centrality

import math

def return_top_n(x, top_n_percent=10):
    top_n = math.ceil((len(x.keys()) * top_n_percent/100))

    return dict(sorted(x.items(), key=lambda item: item[1])[:top_n])



def normalize_if_nan(value):
    x = float(value)

    return 0 if math.isnan(x) else x

NODES_NOs = [100] # number of nodes
rhos, ccls, dens, dgc = list(), list(), list(), list()

configs = {
    0.2: { 
        "popSize": 100,
        "pm": 1,
        "pc": 0.05
    },
    
    0.4: { 
        "popSize": 100,
        "pm": 1,
        "pc": 0.15
    },
    
    0.7: { 
        "popSize": 100,
        "pm": 1,
        "pc": 0.35
    }
}

SCORE_THRESHOLD = 0.2
for nodes_no in NODES_NOs:
    fitnesses = dict()
    cc = 0
    while cc < 10:
        fitnesses.setdefault(cc, {"gen": [], "rho": [], "clustering_coefficient": [], "fitness_score_diff": []})

        print(cc, nodes_no, SCORE_THRESHOLD, flush=True)
        start = time()

        res = {'degree_centrality': [],
            'clustering_coefficient': [],
            'assort_degree': [],
            'density': []}
        
        
        pm = configs[SCORE_THRESHOLD]['pm']
        pc = configs[SCORE_THRESHOLD]['pc']
        popSize = configs[SCORE_THRESHOLD]['popSize']
        evolved_G = start_ga(nodes_no, threshold=SCORE_THRESHOLD, fitnesses=fitnesses[cc], pc=pc, pm=pm, popSize=popSize)

        final_rho, final_ccl, density, degree_centrality = compute_general_metrics("P-test-b", evolved_G, res)

        rhos.append(final_rho)
        ccls.append(final_ccl)
        dens.append(density)
        dgc.append(degree_centrality)

        nodes_map = dict()

        for n in G_evolved.nodes():
            nodes_map[n] = n + 1

        G_evolved = nx.relabel_nodes(G_evolved, nodes_map)

        network_name = "motifs-{}-{}-{}".format(SCORE_THRESHOLD, nodes_no, cc)

        with open(f"results/generated_g-{network_name}.network", "w") as f:
            for e in G_evolved.edges():
                if e[0] == e[1]:
                    continue
                f.write("{}\t{}\t1\n".format(e[0], e[1]))

        print(round(time() - start, 2), flush=True)
        print("----------------------------------------------------------------------------------------------------------------------------------", flush=True)
        cc += 1

