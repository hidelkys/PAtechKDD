from sys import exit, exc_info, argv
import numpy as np
import pandas as pd


from netsapi.challenge import *
from tree_planning_improve import MCTS
from tree_planning_improve import ov


class CustomAgent:
    def __init__(self, environment):
        self.environment = environment
        self.agent = MCTS()

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        rewards = []
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(20):
                print('episode ..................................{}'.format(i))
                self.environment.reset()
                policy = {}
                reward = 0
                for j in range(5):  # episode length
                    action = list(self.agent.get_move(flag=j+1))

                    policy[str(j + 1)] = action
                    s,r,done,info = self.environment.evaluateAction(action)
                    reward += r

                    print('..........year: {}, action: {} , reward {}'.format(j + 1, action,r))
                    self.agent.update(r, done)
                    ov.overall_dicts[j][tuple(action)].append(r)

                print(ov.overall_dicts[0])
                print(ov.overall_dicts[1])
                print(ov.overall_dicts[2])
                print(ov.overall_dicts[3])
                print(ov.overall_dicts[4])

                candidates.append(policy)
                rewards.append(reward)
                # print('max return : ', np.nanmax(rewards))
                print('max return : ', max(rewards))


            # rewards = self.environment.evaluatePolicy(candidates)

            # best_policy = candidates[np.nanargmax(rewards)]
            # best_reward = rewards[np.nanargmax(rewards)]
            best_policy = candidates[np.argmax(rewards)]
            best_reward = rewards[np.argmax(rewards)]

        except (KeyboardInterrupt, SystemExit):
            print(exc_info())

        return best_policy, best_reward

if __name__=='__main__':
    # EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, "sequential_bayesian.csv")
    EvaluateChallengeSubmission(ChallengeProveEnvironment, CustomAgent, "sequential_bayesian_improve_prove_0.99_3000_0.5_11_again.csv")

