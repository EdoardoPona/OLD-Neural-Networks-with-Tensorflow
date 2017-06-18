"""A basic genetic algorithm that produces a known string """
from __future__ import division
import random
import string
import numpy as np

string_values = string.lowercase + ' '


def random_string(length):
    """ returns a random string """
    return ''.join(random.choice(string_values) for i in xrange(length))


def fitness(s, obj):
    """returns the fitness score of a string 's' with regard to an objective string 'obj' """
    fit_val = 0
    for i in xrange(len(obj)):
        if s[i] == obj[i]:
            fit_val += 1
    return fit_val


def population_fitness(pop, obj):
    """returns the fitness score of a population of strings 'pop' with regard to an objective string 'obj' """
    fitness_values = []

    # calculating all the fitness values
    for i in pop:
        fit_val = fitness(i, obj)
        fitness_values.append(fit_val)

    fitness_total = sum(fitness_values)

    # normalizing the fitness_values
    if fitness_total == 0:
        fitness_values = 1 / len(fitness_values)
    else:
        fitness_values = np.array(fitness_values) / fitness_total

    return list(fitness_values)


def mix(mates):
    """ returns the a mix of 'n' mating strings """
    result = ''
    for i in xrange(len(mates[0])):
        j = random.randint(0, len(mates)-1)
        result += mates[j][i]

    return result


def mutate(s, n):
    """ randomly replace 'n' number of characters in the string 's' """
    for i in xrange(n):
        s.replace(s[random.randint(0, len(s)-1)], random.choice(string_values))

    return s

objective = 'hello this is finally working'
best_individual = ''
population_length = 500
population = [random_string(len(objective)) for i in xrange(population_length)]       # we assume we already know the length
fitness_values = population_fitness(population, objective)
new_population = []

generation = 0

while best_individual != objective:

    while len(new_population) < population_length + 1:
        mates = np.random.choice(population, 5, False, fitness_values)
        new_population.append(mutate(mix(mates), 1))

    population = new_population
    new_population = []
    fitness_values = population_fitness(population, objective)

    best_individual = population[fitness_values.index(max(fitness_values))]

    if generation % 20 == 0 or best_individual == objective:
        print 'Generation: ', generation,\
            ' String: ', best_individual,\
            'Fitness: ', fitness(best_individual, objective) / len(objective)

    generation += 1
