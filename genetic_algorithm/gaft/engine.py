#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Genetic Algorithm engine definition
'''

import logging
import math
from functools import wraps

# Imports for profiling.
import cProfile
import pstats
import os

from .components import IndividualBase, Population
from .plugin_interfaces.operators import Selection, Crossover, Mutation
from .plugin_interfaces.analysis import OnTheFlyAnalysis
from .mpiutil import MPIUtil

from keras import models
from keras import layers
import numpy as np

mpi = MPIUtil()

def do_profile(filename, sortby='tottime'):
    ''' Constructor for function profiling decorator.
    '''
    def _do_profile(func):
        ''' Function profiling decorator.
        '''
        @wraps(func)
        def profiled_func(*args, **kwargs):
            '''
            Decorated function.
            '''
            # Flag for doing profiling or not.
            DO_PROF = os.getenv('PROFILING')

            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func

    return _do_profile


class StatVar(object):
    def __init__(self, name):
        ''' Descriptor for statistical variables which need to be memoized when
        engine running.
        '''
        # Protected.
        self.name = '_{}'.format(name)

    def __get__(self, engine, cls):
        '''
        Getter.
        '''
        stat_var = getattr(engine, self.name)
        if stat_var is None:
            if 'min' in self.name and 'ori' in self.name:
                stat_var = engine.population.min(engine.ori_fitness)
            elif 'min' in self.name:
                stat_var = engine.population.min(engine.fitness)
            elif 'max' in self.name and 'ori' in self.name:
                stat_var = engine.population.max(engine.ori_fitness)
            elif 'max' in self.name:
                stat_var = engine.population.max(engine.fitness)
            elif 'mean' in self.name and 'ori' in self.name:
                stat_var = engine.population.mean(engine.ori_fitness)
            elif 'mean' in self.name:
                stat_var = engine.population.mean(engine.fitness)
            setattr(engine, self.name, stat_var)
        return stat_var

    def __set__(self, engine, value):
        '''
        Setter.
        '''
        setattr(engine, self.name, value)


class GAEngine(object):
    ''' Class for representing a Genetic Algorithm engine. The class is the 
    central object in GAFT framework for running a genetic algorithm optimization.
    Once the population with individuals,  a set of genetic operators and fitness 
    function are setup, the engine object unites these informations and provide 
    means for running a genetic algorthm optimization.

    :param population: The Population to be reproduced in evolution iteration.
    :type population: :obj:`gaft.components.Population`

    :param selection: The Selection to be used for individual seleciton.
    :type selection: :obj:`gaft.plugin_interfaces.operators.Selection`

    :param crossover: The Crossover to be used for individual crossover.
    :type crossover: :obj:`gaft.plugin_interfaces.operators.Crossover`

    :param mutation: The Mutation to be used for individual mutation.
    :type mutation: :obj:`gaft.plugin_interfaces.operators.Mutation`

    :param fitness: The fitness calculation function for an individual in population.
    :type fitness: function

    :param analysis: All analysis class for on-the-fly analysis.
    :type analysis: :obj:`OnTheFlyAnalysis` list
    '''
    # Statistical attributes for population.
    fmax, fmin, fmean = StatVar('fmax'), StatVar('fmin'), StatVar('fmean')
    ori_fmax, ori_fmin, ori_fmean = (StatVar('ori_fmax'),
                                     StatVar('ori_fmin'),
                                     StatVar('ori_fmean'))

    def __init__(self, population, selection, crossover, mutation,
                 fitness=None, analysis=None):
        # Set logger.
        logger_name = 'gaft.{}'.format(self.__class__.__name__)
        self.logger = logging.getLogger(logger_name)

        # Attributes assignment.
        self.population = population
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.analysis = [] if analysis is None else [a() for a in analysis]

        # Maxima and minima in population.
        self._fmax, self._fmin, self._fmean = None, None, None
        self._ori_fmax, self._ori_fmin, self._ori_fmean = None, None, None

        # Default fitness functions.
        self.ori_fitness = None if self.fitness is None else self.fitness

        # Store current generation number.
        self.current_generation = -1  # Starts from 0.

        # Check parameters validity.
        self._check_parameters()

    @do_profile(filename='gaft_run.prof')
    def run(self, ng=100):
        ''' Run the Genetic Algorithm optimization iteration with specified parameters.

        :param ng: Evolution iteration steps (generation number)
        :type ng: int
        '''
        if self.fitness is None:
            raise AttributeError('No fitness function in GA engine')
        
        self._update_statvars()

        # Setup analysis objects.
        for a in self.analysis:
            a.setup(ng=ng, engine=self)

        # Enter evolution iteration.
        try:
            for g in range(ng):
                self.current_generation = g

                # The best individual in current population. 
                if mpi.is_master:
                    best_indv = self.population.best_indv(self.fitness)
                else:
                    best_indv = None
                best_indv = mpi.bcast(best_indv)

                # Scatter jobs to all processes.
                local_indvs = []
                # NOTE: One series of genetic operation generates 2 new individuals.
                local_size = mpi.split_size(self.population.size // 2)
                # LSTM
                #model_trained = self.train_LSTM(good_p, bad_p)
                model_trained = None
                # Fill the new population.
                for _ in range(local_size):
                    # Select father and mother.
                    parents = self.selection.select(self.population, fitness=self.fitness)
                    parents = self.formalizaion([parents[0], parents[1]])
                    # Crossover.
                    children = self.crossover.cross(*parents)
                    # Mutation.
                    children = [self.mutation.mutate(child, self) for child in children]
                    children = self.formalizaion(children)
                    kill_number = 0
                    # Generating rules that can be defined
                    while not self.is_qualified(children, model_trained):

                        kill_number += 1
                        print('**************************kill{}******************'.format(kill_number))
                        # Crossover.
                        children = self.crossover.cross(*parents)
                        # Mutation.
                        children = [self.mutation.mutate(child, self) for child in children]
                        children = self.formalizaion(children)
                    # Collect children.
                    local_indvs.extend(children)

                # Gather individuals from all processes.
                indvs = mpi.merge_seq(local_indvs)
                # Retain the previous best two individual.
                indvs[0] = best_indv[0]
                #indvs[1] = best_indv[1]
                # The next generation.
                self.population.individuals = indvs

                # Update statistic variables.
                self._update_statvars()

                # Run all analysis if needed.
                for a in self.analysis:
                    if g % a.interval == 0:
                        a.register_step(g=g, population=self.population, engine=self)
        except Exception as e:
            # Log exception info.
            if mpi.is_master:
                msg = '{} exception is catched'.format(type(e).__name__)
                self.logger.exception(msg)
            raise e

        finally:
            # Recover current generation number.
            self.current_generation = -1
            # Perform the analysis post processing.
            for a in self.analysis:
                a.finalize(population=self.population, engine=self)

    def formalizaion(self, child_list):
        for child in child_list:
            for i in range(len(child.chromsome)):
                child.chromsome[i] = float('%.1f'%child.chromsome[i])
        return child_list

    def train_LSTM(self, tmp_true, tmp_false):
        # LSTM
        def build_model():
            model = models.Sequential()
            model.add(layers.Dense(16,activation='relu',input_shape=(11,)))
            model.add(layers.Dense(16,activation='relu'))
            model.add(layers.Dense(1,activation='sigmoid'))
            model.compile(optimizer='rmsprop',# 还可以通过optimizer = optimizers.RMSprop(lr=0.001)来为优化器指定参数
                        loss='binary_crossentropy', # 等价于loss = losses.binary_crossentropy
                        metrics=['accuracy']) # 等价于metrics = [metircs.binary_accuracy]
            return model
        model = build_model()

        train_data = []
        train_label = []
         
        # True data
        for d in tmp_true:
            tmp = []
            for v in list(d.values()):
                v = [int(v[i] * 10) for i in range(len(v))]
                tmp.extend(v)
            train_label.append(1)
            train_data.append((list(tmp)))

        # False data    
        for d in tmp_false:
            tmp = []
            for v in list(d.values()):
                v = [int(v[i] * 10) for i in range(len(v))]
                tmp.extend(v)
            train_label.append(0)
            train_data.append((list(tmp)))

        train_data = self.vectorize_sequences(train_data)

        if len(train_data) != 0:
            random_seed = list(range(len(train_data)))
            np.random.shuffle(random_seed)
            train_data_ = np.zeros((len(train_data), 11))
            train_label_ = np.zeros((len(train_label), 1))
            
            for i in range(len(random_seed)):
                #print('train_data_',train_data_[random_seed[i]])
                #print('train_data',train_data[i])
                train_data_[random_seed[i]] = train_data[i]
                train_label_[random_seed[i]] = train_label[i]

            train_data = train_data_
            train_label = train_label_

        train_label = np.asarray(train_label).astype('float32')
        train_data = np.array(train_data)

        cut = int(len(train_data) * 0.3)
        x_val = train_data[:cut]
        partial_x_train = train_data[cut:]
        y_val = train_label[:cut]
        partial_y_train = train_label[cut:]

        history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20, # 在全数据集上迭代40次
                    batch_size=3) # 每个batch的大小为1
                    #validation_data=(x_val,y_val))
        return model

    # encode function
    def vectorize_sequences(self, sequences,dimension=11): # input nparray
        results = np.zeros((len(sequences),dimension))
        # one-hot encoding
        for i,sequence in enumerate(sequences):
            #print(sequence, type(sequence), 'sequencesequence')
            results[i,[int(s) for s in sequence]] = 1.
        return results


    def is_qualified(self, children, model):
        child1 = children[0]
        child2 = children[1]
        low = child1.actions[0]
        high = child1.actions[1]
        valid_actions = []
        for l in low:
            for h in high:
                valid_actions.append([l, h])
                valid_actions.append([h, l])

        model_trained = model

        def helper(child, valid_actions, model):
            # check whether the childs' chromesome is valid for the rules
            child = child.chromsome
            # flag = -1 if child[0] >= child[1] else 1
            # for i in range(2, len(child), 2):
            #     flag_tmp = -1 if child[i] >= child[i + 1] else 1
            #     if flag_tmp != -flag:
            #         print('jiaoti') # just for check
            #         return False
            #     flag = -flag
            
            # make the gap as big as possible
            old1 = -1
            old2 = -1
            bad = 0
            min_gap = 0.0
            max_gap = 1.8
            bad_gap = 0.0
            bad_count = 1
            sum_bound = 4.5
            
            for i in range(0, len(child), 2):
                if abs(child[i + 1] - child[i]) <= min_gap or abs(child[i + 1] + child[i]) > max_gap:
                    print('gaAAAAp', [child[i], child[i + 1]], child)
                    return False
                if abs(child[i] - old1) <= bad_gap or abs(child[i + 1] - old2) <= bad_gap:
                    bad += 1
                old1 = child[i]
                old2 = child[i + 1]
            
            if sum(child[2:]) >= sum_bound:
                return False
            if bad > bad_count:
                print('BAD')
                return False
            
            # make sure the inner order of actions, for example 1,0 0,1 or 0,1 1,0
#            flag = 1 if child[0] > child[1] else -1
#            for i in range(2, len(child), 2):
#                tmp_flag = 1 if child[i] > child[i + 1] else -1
#                flag = -flag
#                if tmp_flag != flag:
#                    return False

            # check whether the childs' actions in the valid action space
            # for i in range(0, len(child), 2):
            #     if [child[i], child[i + 1]] not in valid_actions:
            #         print('Wrong actions', (child[i], child[i + 1]), valid_actions)
            #         return False


            # LSTM
            # if self.current_generation >= 1:
            #     tmp = [int(child[i] * 10) for i in range(len(child))]
            #     input_child = self.vectorize_sequences(np.array([tmp]))
            #     prediction = model.predict(input_child)[0][0]
            #     print(prediction)
            #     if prediction > 0.4:
            #         return True
            #     if prediction <= 0.4:
            #         print('*******LSTM kill******')
            #         return False

            return True

        return helper(child1, valid_actions, model_trained) and helper(child2, valid_actions, model_trained)
                    

        
    def _update_statvars(self):
        '''
        Private helper function to update statistic variables in GA engine, like
        maximum, minimum and mean values.
        '''
        # Wrt original fitness.
        self.ori_fmax = self.population.max(self.ori_fitness)
        self.ori_fmin = self.population.min(self.ori_fitness)
        self.ori_fmean = self.population.mean(self.ori_fitness)

        # Wrt decorated fitness.
        self.fmax = self.population.max(self.fitness)
        self.fmin = self.population.min(self.fitness)
        self.fmean = self.population.mean(self.fitness)

    def _check_parameters(self):
        '''
        Helper function to check parameters of engine.
        '''
        if not isinstance(self.population, Population):
            raise TypeError('population must be a Population object')
        if not isinstance(self.selection, Selection):
            raise TypeError('selection operator must be a Selection instance')
        if not isinstance(self.crossover, Crossover):
            raise TypeError('crossover operator must be a Crossover instance')
        if not isinstance(self.mutation, Mutation):
            raise TypeError('mutation operator must be a Mutation instance')

        for ap in self.analysis:
            if not isinstance(ap, OnTheFlyAnalysis):
                msg = '{} is not subclass of OnTheFlyAnalysis'.format(ap.__name__)
                raise TypeError(msg)

    # Decorators.

    def fitness_register(self, fn):
        ''' A decorator for fitness function register.

        :param fn: Fitness function to be registered
        :type fn: function
        '''
        @wraps(fn)
        def _fn_with_fitness_check(indv):
            '''
            A wrapper function for fitness function with fitness value check.
            '''
            # Check indv type.
            if not isinstance(indv, IndividualBase):
                raise TypeError('indv\'s class must be subclass of IndividualBase')

            # Check fitness.
            fitness = fn(indv)
            is_invalid = (type(fitness) is not float) or (math.isnan(fitness))
            if is_invalid:
                msg = 'Fitness value(value: {}, type: {}) is invalid'
                msg = msg.format(fitness, type(fitness))
                raise ValueError(msg)
            return fitness

        self.fitness = _fn_with_fitness_check
        if self.ori_fitness is None:
            self.ori_fitness = _fn_with_fitness_check

    def analysis_register(self, analysis_cls):
        ''' A decorator for analysis regsiter.

        :param analysis_cls: The analysis to be registered
        :type analysis_cls: :obj:`gaft.plugin_interfaces.OnTheFlyAnalysis`
        '''
        if not issubclass(analysis_cls, OnTheFlyAnalysis):
            raise TypeError('analysis class must be subclass of OnTheFlyAnalysis')

        # Add analysis instance to engine.
        analysis = analysis_cls()
        self.analysis.append(analysis)

    # Functions for fitness scaling.

    def linear_scaling(self, target='max', ksi=0.5):
        '''
        A decorator constructor for fitness function linear scaling.

        :param target: The optimization target, maximization or minimization,
                       possible value: 'max', 'min'
        :type target: str

        :param ksi: Selective pressure adjustment value.
        :type ksi: float

        .. Note::

            Linear Scaling:
                1. :math:`arg \max f(x)`, then the scaled fitness would be :math:`f - \min f(x) + {\\xi}`
                2. :math:`arg \min f(x)`, then the scaled fitness would be :math:`\max f(x) - f(x) + {\\xi}`

        '''
        def _linear_scaling(fn):
            # For original fitness calculation.
            self.ori_fitness = fn

            @wraps(fn)
            def _fn_with_linear_scaling(indv):
                # Original fitness value.
                f = fn(indv)

                # Determine the value of a and b.
                if target == 'max':
                    f_prime = f - self.ori_fmin + ksi
                elif target == 'min':
                    f_prime = self.ori_fmax - f + ksi
                else:
                    raise ValueError('Invalid target type({})'.format(target))
                return f_prime

            return _fn_with_linear_scaling

        return _linear_scaling

    def dynamic_linear_scaling(self, target='max', ksi0=2, r=0.9):
        '''
        A decorator constructor for fitness dynamic linear scaling.

        :param target: The optimization target, maximization or minimization
                       possible value: 'min' or 'max'
        :type target: str

        :param ksi0: Initial selective pressure adjustment value, default value is 2
        :type ksi0: float

        :param r: The reduction factor for selective pressure adjustment value,
                  ksi^(k-1)*r is the adjustment value for generation k, default
                  value is 0.9
        :type r: float in range [0.9, 0.999]

        .. Note::
            Dynamic Linear Scaling:

            For maximizaiton, :math:`f' = f(x) - \min f(x) + {\\xi}^{k}`, :math:`k` is generation number.
        '''
        def _dynamic_linear_scaling(fn):
            # For original fitness calculation.
            self.ori_fitness = fn

            @wraps(fn)
            def _fn_with_dynamic_linear_scaling(indv):
                f = fn(indv)
                k = self.current_generation + 1

                if target == 'max':
                    f_prime = f - self.ori_fmin + ksi0*(r**k)
                elif target == 'min':
                    f_prime = self.ori_fmax - f + ksi0*(r**k)
                else:
                    raise ValueError('Invalid target type({})'.format(target))
                return f_prime

            return _fn_with_dynamic_linear_scaling

        return _dynamic_linear_scaling

    def minimize(self, fn):
        ''' A decorator for minimizing the fitness function.

        :param fn: Original fitness function
        :type fn: function
        '''
        @wraps(fn)
        def _minimize(indv):
            return -fn(indv)
        return _minimize

