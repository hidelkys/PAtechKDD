#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


gamma = 0.99
c_puct = 3000
discrete_num = 11
overall_ratio = 0.5  # overall和own加权

# c_puct = gl.get_value('c_puct')
# discrete_num = gl.get_value('discrete_num')
# overall_ratio = gl.get_value('overall_ratio')

class OverallValue(object):
    def __init__(self):
        self.overall_dicts = [dict(), dict(), dict(), dict(), dict()]
        for dic in self.overall_dicts:
            for i in np.arange(discrete_num) / (discrete_num - 1):
                for j in np.arange(discrete_num) / (discrete_num - 1):
                    dic[(i, j)] = []

    def reset(self):
        self.overall_dicts = [dict(), dict(), dict(), dict(), dict()]
        for dic in self.overall_dicts:
            for i in np.arange(discrete_num) / (discrete_num - 1):
                for j in np.arange(discrete_num) / (discrete_num - 1):
                    dic[(i, j)] = []
ov = OverallValue()

def policy_value_fn_random():
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(discrete_num*discrete_num)/ (discrete_num*discrete_num)
    actions = []
    for i in np.arange(discrete_num)/(discrete_num-1):
        for j in np.arange(discrete_num)/(discrete_num-1):
            actions.append((i,j))
    # zip -> <zip object at 0x00000240E1046E88>
    return zip(actions, action_probs)

def policy_value_fn_dirichlet():
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.random.dirichlet(1. * np.ones(discrete_num*discrete_num))
    actions = []
    for i in np.arange(discrete_num)/(discrete_num-1):
        for j in np.arange(discrete_num)/(discrete_num-1):
            actions.append((i,j))
    # zip -> <zip object at 0x00000240E1046E88>
    return zip(actions, action_probs)

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent # 父节点
        self._children = {} # 所有子节点
        self._n_visits = 0 # 当前节点访问次数
        self._Q = 0 # value，注意这里的value不再是0-1之间了，到时候c_puct要好好调整
        self._r = 0 # reward,中间变量，用来算value
        self._u = 0 # 探索度
        self._P = prior_p # 先验概率
        self._flag = 0 # 加一个标志位，看当前树到第几层了

    def expand(self, action_priors,flag):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
                self._children[action]._flag = flag

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        own_value = dict()
        overall_value = dict()# 这个地方分开算的，其实不用算own,可以写到一起算，后期再优化
        for key,node in self._children.items():
            own_value[key] = node.get_value(c_puct)

        for key,value in own_value.items():
            if ov.overall_dicts[self._flag][key] != []:
                overall_value[key] = overall_ratio*np.mean(ov.overall_dicts[self._flag][key])+(1-overall_ratio)*own_value[key]
            else:
                overall_value[key] = own_value[key]

        move = max(overall_value,key=overall_value.get)
        child = self._children[move]

        return move,child
        # return max(self._children.items(),
        #            key=lambda act_node: act_node[1].get_value(c_puct))
        # 这里self._children是一个字典，键是动作，键值是下一个节点。
        # act_node[1].get_value就是看这个动作之后的那个状态节点的值，就是Q+u，然后返回值最大的动作和对应到达的节点。
        # 这里的get_values是后面定义的函数，不是dict自带的

    def update(self,reward):
        self._n_visits += 1
        self._r = reward


    def backup(self):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        self._Q += 1.0 * (self._r - self._Q) / self._n_visits
        # 这个Q是平均胜率，每次更新就把这次的value平均到n_visits上，再加上之前的就好了。
        # 这里其实就是(v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)

    def backup_recursive(self):
        if self._parent:
            self._parent._r += self._r * gamma
            self._parent.backup_recursive()
        self.backup()

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):

    def __init__(self, policy_value_fn=policy_value_fn_dirichlet,c_puct=c_puct):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._root._n_visits = 1
        #这里None 和 1分别赋值的parent, prior_p，
        # 因为刚开始这是根节点，所以没有父节点，到达这个节点的概率肯定是1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self.current_node = self._root

    def get_move(self,flag):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        # if self.current_node.is_root:
        #     self.current_node._n_visits += 1

        if self.current_node.is_leaf():
            action_priors = self._policy()
            self.current_node.expand(action_priors=action_priors,flag=flag)

        move,self.current_node = self.current_node.select(c_puct=self._c_puct)

        return move

    def update(self,reward,done):
        self.current_node.update(reward)

        if done:
            self.current_node.backup_recursive()
            self.current_node = self._root

    def __str__(self):
        return "MCTS"

if __name__ == '__main__':
    mcts = MCTS(policy_value_fn=policy_value_fn_random)

    i = 0
    action = mcts.get_move(flag=i+1)
    mcts.update(5, False)
    ov.overall_dicts[i][(action)].append(5)
    i = 1
    action = mcts.get_move(flag=i)
    mcts.update(4, False)
    ov.overall_dicts[i][(action)].append(4)
    i = 2
    action = mcts.get_move(flag=i+1)
    mcts.update(3, False)
    ov.overall_dicts[i][(action)].append(3)
    i = 3
    action = mcts.get_move(flag=i+1)
    mcts.update(2, False)
    ov.overall_dicts[i][(action)].append(2)
    i = 4
    action = mcts.get_move(flag=i+1)
    mcts.update(1, True)
    ov.overall_dicts[i][(action)].append(1)

