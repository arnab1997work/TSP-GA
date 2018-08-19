
===========================================================================================================
##TSP-Cover all cities via the shortest possible route-cover all cities once and return to travelling city.
===========================================================================================================

import numpy as np

import random

import operator

import pandas as pd

import matplotlib.pyplot as pyplot

 

class City:
    def __init__(self, x,y):
        self.x=x
        self.y=y
    def distance(self, city):
        x_distance=abs(self.x-city.x)
        y_distance=abs(self.y-city.y)
        distance=np.sqrt((x_distance**2)+(y_distance**2))
        return distance
    def __repr__(self):
        return "(" +str(self.x) +"," +str(self.y)+ ")"
    

class strength:
    def __init__(self, route):
        self.route=route
        self.distance=0
        self.fitness=0.0
    def route_distance(self):
        if self.distance ==0:
            pathDistance=0
            for i in range(0,len(self.route)):
                sourceCity=self.route[i]
                destCity=None
                if i+1<len(self.route):     ## travelling through all cities
                    destCity=self.route[i+1]
                else:
                    destCity=self.route[0]  ## starting and end points must be same
                pathDistance+= destCity.distance(sourceCity)
            self.distance= pathDistance
        return pathDistance
    def route_choice(self):
        if self.fitness==0:
            self.fitness =1/float(self.route_distance())  ## return the path choice as inverse of route distance
        return self.fitness
    


"""
random.sample?
Signature: random.sample(population, k)
Docstring:
Chooses k unique random elements from a population sequence or set.

Returns a new list containing elements from the population while
leaving the original population unchanged.  The resulting list is
in selection order so that all sub-slices will also be valid random
samples.  This allows raffle winners (the sample) to be partitioned
into grand prize and second place winners (the subslices).

Members of the population need not be hashable or unique.  If the
population contains repeats, then each occurrence is a possible
selection in the sample.

To choose a sample in a range of integers, use range as an argument.
This is especially fast and space efficient for sampling from a
large population:   sample(range(10000000), 60)



ls=random.sample(range(1,10), 6)
Out[17]: [8, 9, 3, 4, 5, 1]
ls
Out[18]: [8, 9, 3, 4, 5, 1]


ls=random.sample([7,7,777,7,7,43,34,1,45,36,423,4], 8)
Out[27]: [36, 423, 34, 7, 7, 43, 777, 45]

"""




## Create the first set of Solution(population)
===============================================


def createRoute(cityList):
    route=random.sample(cityList, len(cityList))  
    return route




def initialPopulation(populationSize, cityList):
    population=[]
    for i in range(0, populationSize):
        population.append(createRoute(cityList))    ##Create as many routes as the population size
    return population





def rankRoutes(population):
    fitness_values={}
    for i in range(0, len(population)):
        fitness_values[i]=strength(population[i]).route_choice() ## Calulating fitness values of each route sampled
    return sorted(fitness_values.items(), key=operator.itemgetter(1), reverse=True)




def selection(populationRanked, eliteSize):
    selectionResults=[]
    df= pd.DataFrame(np.array(populationRanked),columns=['Index','Fitness'])
    df['cum_sum']= df.Fitness.cumsum()
    df['cum_percentage']=100*df.cum_sum/df.Fitness.sum()

    for i in range(0,eliteSize):
        selectionResults.append(populationRanked[i][0]) ##Chose all the elites from the ranked population list

    for i in range(0, len(populationRanked)-eliteSize):
        pick=100*random.random()
        for i in range(0, len(populationRanked)):
            if pick< df.iat[i,3]:                       ##Return the value at position [i,3]
                selectionResults.append(populationRanked[i][0])
                break
    return selectionResults



def mating_pool(population, selectionResults):
    matingPool=[]
    for i in range(0,len(selectionResults)):
        index=selectionResults[i]
        matingPool.append(population[index]) ##Returns solutions according to index in selectionResults
    return matingPool




def breed(parent1, parent2):
    child=[]
    childP1=[]
    childP2=[]

    geneA=int(random.random()*len(parent1))
    geneB=int(random.random()*len(parent1))

    startGene=min(geneA, geneB)
    endGene=max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2=[item for item in parent2 if item not in childP1]  ##Ordered-crossover, take from parent 2 items that aren't already there in the Child 1
    child=childP1+childP2

    return child




def breed_population(matingPool, eliteSize):
    children=[]
    length= len(matingPool)-eliteSize
    pool=random.sample(matingPool, len(matingPool))

    for i in range(0, eliteSize):
        children.append(matingPool[i])
    for i in range(0, length):
        child=breed(pool[i], pool[len(matingPool)-i-1]) ## choose parents from first and last to mate
        children.append(child)
    return children



def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random()<mutationRate):
            swapWith=int(random.random())*len(individual)

            city1=individual[swapped]
            city2=individual[swapWith]

            individual[swapped]=city2
            individual[swapWith]=city1

    return individual





def mutatePopulation(population, mutationRate):
    mutatedPopulation=[]

    for ind in range(0, len(population)):
        mutatedIndividual= mutate(population[ind], mutationRate)
        mutatedPopulation.append(mutatedIndividual)

    return mutatedPopulation




def next_gen(currentGen, eliteSize, mutationRate):
    populationRanked=rankRoutes(currentGen)
    selectionResults=selection(populationRanked, eliteSize)
    matingPool=mating_pool(currentGen, selectionResults)
    children=breed_population(matingPool,eliteSize)
    nextGeneration=mutatePopulation(children, mutationRate)
    return nextGeneration



def GA(population, populationSize, eliteSize, mutationRate, generations):
    pop=initialPopulation(populationSize, population)
    print("Initial Distance: " +str(1/ rankRoutes(pop)[0][1]))
    for i in range(0, generations):
        pop=next_gen(pop, eliteSize, mutationRate)

    print("Final Distance: " +str(1/rankRoutes(pop)[0][1]))
    bestRouteIndex=rankRoutes(pop)[0][0]
    bestRoute=pop[bestRouteIndex]
    return bestRoute



def GAplot(population, populationSize, eliteSize, mutationRate, generations):
    pop=initialPopulation(populationSize, population)
    progress=[]
    progress.append(1/rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop= next_gen(pop, eliteSize, mutationRate)
        progress.append(1/rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel("Distance")
    plt.xlabel("Generation")
    plt.show()







cityList=[]

for i in range(0,25):
    cityList.append(City(x=int(random.random()*200), y=int(random.random()*200)))
    

GA(population=cityList, populationSize=100, eliteSize=20, mutationRate=0.01, generations=500)

Initial Distance: 1734.1247631517726
Final Distance: 840.2431420717323
Out[67]: 
[(103,27),
 (103,19),
 (98,13),
 (70,39),
 (27,4),
 (29,27),
 (53,64),
 (58,104),
 (63,135),
 (75,125),
 (91,81),
 (126,50),
 (147,54),
 (182,49),
 (162,83),
 (190,144),
 (119,168),
 (117,179),
 (83,176),
 (135,122),
 (130,102),
 (136,48),
 (135,45),
 (138,41),
 (127,28)]

