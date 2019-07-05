#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Flip Bit mutation implementation. '''

from random import random, uniform
import random as rd # avoid conflict

from ...mpiutil import MPIUtil
from ...plugin_interfaces.operators.mutation import Mutation
from ...components.binary_individual import BinaryIndividual
from ...components.decimal_individual import DecimalIndividual
from ...components.order_individual import OrderIndividual

mpi = MPIUtil()


class FlipBitMutation(Mutation):
    ''' Mutation operator with Flip Bit mutation implementation.

    :param pm: The probability of mutation (usually between 0.001 ~ 0.1)
    :type pm: float in range (0.0, 1.0]
    '''
    def __init__(self, pm):
        if pm <= 0.0 or pm > 1.0:
            raise ValueError('Invalid mutation probability')

        self.pm = pm

    def mutate(self, individual, engine):
        ''' Mutate the individual.

        :param individual: The individual on which crossover operation occurs
        :type individual: :obj:`gaft.components.IndividualBase`

        :param engine: Current genetic algorithm engine
        :type engine: :obj:`gaft.engine.GAEngine`

        :return: A mutated individual
        :rtype: :obj:`gaft.components.IndividualBase`
        '''
        do_mutation = True if random() <= self.pm else False
        
        if do_mutation:
            # Only for OrderIndividual
            low = individual.actions[0]
            high = individual.actions[1]
            flag_forOI = rd.choice(['odd','even'])
            if flag_forOI == 'odd':
                pass
            else:
                low, high = high, low
            
            flag = -1
            index = 0
            # Only for OrderIndividual
            for i, genome in enumerate(individual.chromsome):
                no_flip = True if random() > self.pm else False
                if no_flip:
                    continue
                if type(individual) is BinaryIndividual:
                    individual.chromsome[i] = genome^1
                elif type(individual) is DecimalIndividual:
                    a, b = individual.ranges[i]
                    eps = individual.precisions[i]
                    n_intervals = (b - a)//eps
                    n = int(uniform(0, n_intervals + 1))
                    individual.chromsome[i] = a + n*eps

                elif type(individual) is OrderIndividual:
                    if index % 2 == 1: # change by 2 steps
                        flag = - flag
                    a, b = individual.ranges[i]
                    tmp = rd.choice(low) if flag == 1 else rd.choice(high)
                    individual.chromsome[index] = a + tmp
                    p = individual.chromsome
                    individual.solution = p

                    individual.chromsome[0] = individual.best_1[0]
                    individual.chromsome[1] = individual.best_1[1]
                    # individual.chromsome[2] = 0.6 
                    # individual.chromsome[3] = 0.0
                    # individual.chromsome[4] = 0.4 
                    # individual.chromsome[5] = 0.0
                    # individual.chromsome[6] = 0.8 
                    # individual.chromsome[7] = 0.8
                    #individual.chromsome[8] = 0.1
                    #individual.chromsome[9] = 0.5
                    
                else:
                    raise TypeError('Wrong individual type: {}'.format(type(individual)))
                index += 1

            return individual
            #print(individual.chromsome)
            # Update solution.
            #individual.solution = individual.decode()

        return individual


class FlipBitBigMutation(FlipBitMutation):
    ''' Mutation operator using Flip Bit mutation implementation with adaptive
    big mutation rate to overcome premature or local-best solution.

    :param pm: The probability of mutation (usually between 0.001 ~ 0.1)
    :type pm: float in (0.0, 1.0]

    :param pbm: The probability of big mutation, usually more than 5 times
                bigger than pm.
    :type pbm: float

    :param alpha: intensive factor
    :type alpha: float, in range (0.5, 1)
    '''
    def __init__(self, pm, pbm, alpha):
        super(self.__class__, self).__init__(pm)

        if not (0.0 < pbm < 1.0):
            raise ValueError('Invalid big mutation probability')
        if pbm < 5*pm and mpi.is_master:
            self.logger.warning('Relative low probability for big mutation')
        self.pbm = pbm

        # Intensive factor.
        if not (0.5 < alpha < 1.0):
            raise ValueError('Invalid intensive factor, should be in (0.5, 1.0)')
        self.alpha = alpha

    def mutate(self, individual, engine):
        ''' Mutate the individual with adaptive big mutation rate.

        :param individual: The individual on which crossover operation occurs
        :type individual: :obj:`gaft.components.IndividualBase`

        :param engine: Current genetic algorithm engine
        :type engine: :obj:`gaft.engine.GAEngine`

        :return: A mutated individual
        :rtype: :obj:`gaft.components.IndividualBase`
        '''
        pm = self.pm

        if engine.fmax*self.alpha < engine.fmean:
            self.pm = self.pbm

        # Mutate with big probability.
        individual = super(self.__class__, self).mutate(individual, engine)

        # Recover probability.
        self.pm = pm

        return individual

