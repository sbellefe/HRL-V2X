import math
import random
import numpy as np
import pickle

from collections import namedtuple, deque

np.random.seed(1)
# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Transition = namedtuple('Transition',
                        ('ep_idx', 'time_idx', 'step_idx', 'state', 'action', 'next_state', 'reward', 'done',
                         'avail_actions', 'avail_actions_next'))


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            if tree_idx < 0:
                raise ValueError('the tree idx is smaller than zero')
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class RB_PER_Memory(object):
    beta = 10.0  # importance-sampling
    decay_factor = 0.6

    def __init__(self, capacity, time_steps, transition=Transition):
        self.Transition = transition
        self.capacity = int(math.floor(capacity / time_steps) * time_steps)
        self.tree = SumTree(self.capacity)
        self.temp_buffer = np.zeros(time_steps, dtype=object)  # for all transitions in the buffer
        self.temp_buffer_pointer = 0
        self.time_steps = time_steps

    def push(self, *args):
        """Save a transition"""
        transition_data = self.Transition(*args)
        self.temp_buffer[self.temp_buffer_pointer] = transition_data
        self.temp_buffer_pointer += 1
        if self.temp_buffer_pointer >= self.time_steps:  # if we store a whole trajectory of MDP
            self.temp_buffer_pointer = 0
            for i in range(self.time_steps):
                if i == self.time_steps - 1:
                    self.tree.add(self.beta, self.temp_buffer[i])  # set beta for end of traj
                else:
                    self.tree.add(1.0, self.temp_buffer[i])  # set 1.0 for others

    def __len__(self):
        return self.tree.data_pointer

    def sample(self, batch_size):
        b_idx, b_memory, priorities = [], [], []
        # print('len(self.tree.data[0])', len(self.tree.data[0]), np.empty((n, len(self.tree.data[0]))))
        pri_seg = self.tree.total_p / batch_size  # priority segment

        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            if data != 0:
                b_idx.append(idx)
                # print('the size of data is', data)
                b_memory.append(data)
                priorities.append(p)

        # print('the size of b_memory is', b_memory)
        b_idx = np.array(b_idx)
        priorities = np.array(priorities)
        """
        Renew of priorities
        """
        num_of_transitions = len(b_memory)
        for i in range(num_of_transitions):
            if priorities[i] > 1.0:
                transition_tree_idx = b_idx[i]
                transition_data_pointer = transition_tree_idx - self.tree.capacity + 1
                transition_data_step_idx = self.tree.data[transition_data_pointer].step_idx
                self.tree.update(transition_tree_idx, 1.0)
                if transition_data_step_idx > 0:
                    pre_transition_data_pointer = transition_data_pointer - 1
                    pre_transition_tree_idx = pre_transition_data_pointer + self.tree.capacity - 1
                    pre_transition_data_step_idx = self.tree.data[pre_transition_data_pointer].step_idx
                    if self.tree.data[pre_transition_data_pointer].step_idx == self.tree.data[
                        transition_data_pointer].step_idx - 1 and \
                            self.tree.data[pre_transition_data_pointer].time_idx == self.tree.data[
                        transition_data_pointer].time_idx:
                        pre_transition_priority = priorities[i]
                        self.tree.update(pre_transition_tree_idx, pre_transition_priority)
                    else:
                        print('wrong pre transition: great!')
                else:
                    # transition_data_step_idx == 0 beginning of this trajectory
                    after_transition_data_pointer = transition_data_pointer + self.time_steps - 1
                    after_transition_tree_idx = after_transition_data_pointer + self.tree.capacity - 1
                    after_transition_data_step_idx = self.tree.data[after_transition_data_pointer].step_idx
                    if self.tree.data[after_transition_data_pointer].step_idx == self.tree.data[
                            transition_data_pointer].step_idx + self.time_steps - 1 and \
                        self.tree.data[after_transition_data_pointer].time_idx == self.tree.data[
                            transition_data_pointer].time_idx:
                        after_transition_priority = max(self.decay_factor * priorities[i], 1.0)
                        self.tree.update(after_transition_tree_idx, after_transition_priority)
                    else:
                        raise ValueError('wrong after transition: great!')
                        print('wrong after transition: great!')

        return self.Transition(*zip(*b_memory))

    def save_to_disk(self, filename):
        """
            Save the memory to a file
        """
        with open(filename, 'wb') as f:
            pickle.dump((self.tree.capacity, self.tree.tree, self.tree.data, self.tree.data_pointer), f)

    def load_from_disk(self, filename):
        with open(filename, 'rb') as f:
            (capacity, tree, data, data_pointer) = pickle.load(f)
            self.tree.capacity = capacity
            self.tree.tree = tree
            self.tree.data = data
            self.tree.data_pointer = data_pointer


class ReplayMemory_single_agent(object):

    def __init__(self, capacity, transition=Transition):
        self.Transition = transition
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return self.Transition(*zip(*transitions))

    def sample_batch(self, sample_indexes):
        transitions = self.memory[sample_indexes]
        return self.Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)

    def save_to_disk(self, filename):
        """
            Save the memory to a file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_from_disk(self, filename):
        """
            Load the memory from a file with validation
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Check the data format is consistent with Tuple Transition
        if isinstance(data, deque) and all(isinstance(item, self.Transition) for item in data):
            self.memory = data
        else:
            print("Loaded data is not in the expected format (deque of Transition)")


Global_Transition = namedtuple('Global_Transition',
                               ('global_state', 'global_action', 'global_next_state', 'global_reward'))


class ReplayGlobalMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Global_Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Global_Transition(*zip(*transitions))

    def sample_batch(self, sample_indexes):
        transitions = self.memory[sample_indexes]
        return Global_Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)

    def save_to_disk(self, filename):
        """
            Save the memory to a file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_from_disk(self, filename):
        """
            Load the memory from a file with validation
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Check the data format is consistent with Tuple Global_Transition
        if isinstance(data, deque) and all(isinstance(item, Global_Transition) for item in data):
            self.memory = data
        else:
            print("Loaded data is not in the expected format (deque of Global_Transition)")


class EnvMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, env):
        """Save a transition"""
        self.memory.append(env)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        """Clears all the experiences in the memory."""
        self.memory.clear()