from gaft import GAEngine
from gaft.components import BinaryIndividual, Population, DecimalIndividual, OrderIndividual
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation, ExponentialRankingSelection, LinearRankingSelection
import math
from random import random, uniform
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from netsapi.challenge import *
import random
import matplotlib.pyplot as plt

import datetime
from sys import exit, exc_info, argv
import numpy as np
import pandas as pd
#!pip install gaft --user --upgrade
#!pip3 install git+https://github.com/slremy/netsapi --user --upgrade

from netsapi.challenge import *

class GeneticAlgorithm:
    def __init__(self, environment):
        self.environment = environment
    
    def generate(self):
        
        import random
        eps = 0.2 # equal to actions space resolution # range/eps
        pop_size = 2
        cross_prob = 0.6
        exchange_prob = 0.7
        mutation_pob = 0.8
        generation = 6
        REWARDS = []
        tmp_reward = []
        tmp_policy = []
        bad_p = []
        good_p = []
        turb = 0

        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            first_action = []
            for a1 in [i/10 for i in range(0,11,2)]:
                for a2 in [i/10 for i in range(0,11,2)]:
                    first_action.append([a1, a2])
            # [0,0] is absolutely bad action
            first_action = first_action[1:]

            action_reward = []
            for i in range(len(first_action)):
                ar = self.environment.evaluateAction(first_action[i])
                self.environment.reset()
                action_reward.append(ar[1])
            # get the best policy for first year
            best_action = first_action[action_reward.index(max(action_reward))]
            print('*****--------------------------', best_action,'--------------------')

            test_action = ([[i/10 for i in range(0, 11, 2)],[i/10 for i in range(0, 11, 2)]])
            
            # 2. Define population
            indv_template = OrderIndividual(ranges=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                                                         eps=eps, actions = test_action, best_1 = best_action) # low_bit and high_bit
            population = Population(indv_template=indv_template, size = pop_size)
            population.init()  # Initialize population with individuals.

            # 3. Create genetic operators

            # Use built-in operators here.
            selection = LinearRankingSelection()

            crossover = UniformCrossover(pc=cross_prob, pe=exchange_prob) # PE = Gene exchange probability
            mutation = FlipBitMutation(pm=mutation_pob) # 0.1 todo The probability of mutation

            # 4. Create genetic algorithm engine to run optimization
            engine = GAEngine(population=population, selection=selection,
                            crossover=crossover, mutation=mutation)
                
            # 5. Define and register fitness function   
            @engine.fitness_register
            def fitness(indv):
                p = [0 for _ in range(10)]
                p = indv.solution
                # encode
                policy = {'1': [p[0], p[1]], '2': [p[2], p[3]], '3': [p[4], p[5]], '4': [p[6], p[7]], '5': [p[8], p[9]]}
                reward = self.environment.evaluatePolicy(policy) 
                tmp_reward.append(reward)
                tmp_policy.append(policy)
                return reward + uniform(-turb, turb)

            @engine.analysis_register
            class ConsoleOutput(OnTheFlyAnalysis):
                master_only = True
                interval = 1
                def register_step(self, g, population, engine):
                    best_indv = population.best_indv(engine.fitness)
                    msg = 'Generation: {}, best fitness: {:.3f}'.format(g + 1, engine.fmax)
                    REWARDS.append(max(tmp_reward[pop_size * (g - 0): pop_size * (g + 1)]))
                    engine.logger.info(msg)
            
            engine.run(ng = generation)
            best_reward = max(tmp_reward)
            best_policy = tmp_policy[-pop_size]
        
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
        
        return best_policy, best_reward

if __name__ == "__main__":
    time_stamp = datetime.datetime.now()
    eps = 0.2
    EvaluateChallengeSubmission(ChallengeProveEnvironment, GeneticAlgorithm, f"kw_7.5_{time_stamp}_{eps}.csv")
    

