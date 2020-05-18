import numpy
import random
import json
import requests
import client_moodle

population_size = 6 #no of parents 
number_of_weights = 11

def gen_mating_pool_probab(fitness) :
    probabilities = list(numpy.reciprocal(fitness) / float(sum(numpy.reciprocal(fitness))))
    return probabilities

def select_mates(probabs,genes):
    probabs.sort(reverse = True)
    if random.randint(0, 3):
        parent_index = numpy.random.choice(population_size, 2, probabs) #selects 2 parents based on the generated probabilities
        parents = [genes[parent_index[0]], genes[parent_index[1]]]
    else :
        parent_index = numpy.random.choice(population_size, 1, probabs) #selects 2 parents based on the generated probabilities
        parents = [genes[parent_index[0]], genes[5]]
    return parents

def crossover(parents): #crosses over two parent chromosomes
    child = []
    for i in range(len(parents[0])):
        x = random.randint(0,1)
        child.append(0.9*parents[x][i] + 0.1*parents[1-x][i])
    print("Vector selected for crossover:\nParent 1: {}\nParent 2: {}".format(parents[0],parents[1]))
    return child

def mutate(child):  #adds mutation to a child chromosome #abhi ismein sure nhi h....mtlb if my type doesn't work ..we have to look for other options as well
    print("Child BEFORE Mutation:\n{}".format(child))
    for index in range(11):
        if random.randint(0,1):
            child[index] += 0.1*child[index]
        else:
            child[index] -= 0.1*child[index]
    print("Child AFTER Mutation:\n{}".format(child))
    return child

def find_mean_error(training_error, validation_error) : # #ensuring ki ham validation error ko bhi acchi khaasa weightage de rhe ho
    mean_err = (((training_error * validation_error * 1.3) / 10000000) + (training_error + validation_error / 9)) / 2
    return mean_err

with open('input_my.json','r') as f:
    parent_error = json.load(f) #parent_error is a list which contains lists of[[error,parent],[error,parent],[error,parent]]

chromosomes = list(numpy.asarray(parent_error)[:,3])
fitness = list(numpy.asarray(parent_error)[:,0])

generations = 15
child_number = 10
min_training_error = parent_error[0][1]
min_validation_error = parent_error[0][2]
min_error = find_mean_error(min_training_error, min_validation_error)

for generation in range(generations):

    print("Generation: {}".format(generation))
    child_errors = []
    chromosome_probability = gen_mating_pool_probab(fitness)

    for i in range(child_number):

        print("\nChild: {}".format(i))
        child = crossover(select_mates(chromosome_probability,chromosomes))
        print("Child: {}".format(child))
        if random.randint(0,1):
            mutate_child = child
        else:
            mutate_child = mutate(child)            
            
        err = client_moodle.get_errors('32EZBpTMqjBF5XByc7riOKvbIw2EykjhBEUqjDSAA9geHjqTaW', list(mutate_child))
        submit_status = client_moodle.submit('32EZBpTMqjBF5XByc7riOKvbIw2EykjhBEUqjDSAA9geHjqTaW', list(mutate_child))

        mean_error = find_mean_error(err[0], err[1])
        min_training_error = min(min_training_error, err[0])
        min_validation_error = min(min_validation_error, err[1])
        min_error = min(min_error, mean_error)
        
        child_errors.append([mean_error, err[0], err[1], child])
    
    random_values = numpy.array(random.sample(range(population_size), 5))

    for index in random_values:
        child_errors.append(parent_error[index])
    child_errors.sort()

    parent_error = child_errors[:population_size-1]
    parent_error.append(child_errors[population_size+2])
    fitness = list(numpy.asarray(parent_error)[:,0])
    
    print("\nGeneration {} \nPopulation : {}".format(generation,parent_error))

with open('input_my.json', 'w') as fjson:
    json.dump(parent_error[:population_size], fjson)