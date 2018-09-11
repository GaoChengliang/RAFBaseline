from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import random
import numpy as np


class IntegerMaze(object):
    def __init__(self, dir, min_integer=0, max_integer=9999, num_each_op=5, len_path=4, num_paths=10, operand_min=1, operand_max=100, random_seed=1111):
        self.dir = dir
        self.min_integer = min_integer
        self.max_integer = max_integer
        self.num_each_op = num_each_op
        self.len_path = len_path
        self.num_paths = num_paths
        self.operand_min = operand_min
        self.operand_max = operand_max

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.ops_pool = self.build_ops_pool(self.num_each_op)
        self.op_paths = self.build_op_paths(self.ops_pool, self.len_path, self.num_paths)

        self.write_entities()
        self.write_relations()
        self.write_facts()
        self.write_data()

    def build_ops_pool(self, size):
        ops_pool = []
        operand_min = self.operand_min
        operand_max = self.operand_max

        pool = []
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('+%d' % i, (x+i - self.min_integer) % (self.max_integer - self.min_integer + 1) + self.min_integer))
        ops_pool.append(random.sample(pool, size))

        pool = []
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('-%d' % i, (x-i - self.min_integer) % (self.max_integer - self.min_integer + 1) + self.min_integer))
        ops_pool.append(random.sample(pool, size))

        pool = []
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('*%d' % i, (x*i - self.min_integer) % (self.max_integer - self.min_integer + 1) + self.min_integer))
        ops_pool.append(random.sample(pool, size))

        pool = []
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('/%d' % i, (int(x/i) - self.min_integer) % (self.max_integer - self.min_integer + 1) + self.min_integer))
        ops_pool.append(random.sample(pool, size))

        return ops_pool

    def build_op_paths(self, ops_pool, len_path, num_paths):
        op_paths = []
        total_ops = np.array(ops_pool).reshape(-1)

        for i in range(num_paths):
            path = []
            for ops in ops_pool:
                path.append(ops[np.random.randint(low=0, high=self.num_each_op)])
            path.extend(random.sample(total_ops, len_path - 4))
            np.random.shuffle(path)
            op_paths.append(path)
        return op_paths

    def path_str(self, path):
        return '_'.join(map(lambda f: f(0)[0], path))

    def training_data(self):
        training_data = []
        for _ in range(self.training_size):
            src = np.random.randint(low=self.min_integer, high=self.max_integer + 1)
            dst = src
            i = np.random.randint(low=0, high=self.num_paths)
            path = self.op_paths[i]
            for op in path:
                dst = op(dst)[1]
            training_data.append([src, dst, i])
        return training_data

    def test_data(self):
        test_data = []
        for _ in range(self.test_size):
            src = np.random.randint(low=self.min_integer, high=self.max_integer + 1)
            dst = src
            i = np.random.randint(low=0, high=self.num_paths)
            path = self.op_paths[i]
            for op in path:
                dst = op(dst)[1]
            test_data.append([src, dst, i])
        return test_data

    def write_data(self):
        train_path = self.dir + '/train.txt'
        test_path = self.dir + '/test.txt'

        data = []
        for src in range(self.min_integer, self.max_integer + 1):
            dst = src
            i = np.random.randint(low=0, high=self.num_paths)
            path = self.op_paths[i]
            for op in path:
                dst = op(dst)[1]
            data.append('{}\t{}\t{}\n'.format(src, self.path_str(path), dst))

        np.random.shuffle(data)
        train_data, test_data = data[:int(len(data) * 3 / 4)], data[int(len(data) * 3 / 4):]

        with open(train_path, 'w') as f:
            for line in train_data:
                f.write(line)

        with open(test_path, 'w') as f:
            for line in test_data:
                f.write(line)

    def write_entities(self):
        path = self.dir + '/entities.txt'
        with open(path, 'w') as f:
            for i in range(self.min_integer, self.max_integer + 1):
                    f.write('{}\n'.format(i))

    def write_relations(self):
        path = self.dir + '/relations.txt'
        with open(path, 'w') as f:
            for ops in self.ops_pool:
                for op in ops:
                    f.write('{}\n'.format(op(0)[0]))
            for path in self.op_paths:
                f.write(self.path_str(path) + '\n')

    def write_facts(self):
        path = self.dir + '/facts.txt'
        with open(path, 'w') as f:
            for i in range(self.min_integer, self.max_integer + 1):
                for ops in self.ops_pool:
                    for op in ops:
                        f.write('{}\t{}\t{}\n'.format(i, op(0)[0], op(i)[1]))


def eval_path():
    real_paths = {}

    invs = {
        'inv_*': '/',
        'inv_/': '*',
        'inv_+': '-',
        'inv_-': '+'
    }

    rules = '../exps/maze/rules.txt'
    with open('../datasets/maze/relations.txt', 'r') as path_file:
        paths = path_file.readlines()[20:]
        for path in paths:
            real_paths[path.strip()] = [0, 0, 0, 0]

    print(real_paths)

    with open(rules, 'r') as rules_file:
        for line in rules_file:
            if line.strip().split('\t')[1].startswith('inv_'):
                continue

            info = line.strip().split('\t')[1].split(')')
            real_path = info[0][:info[0].find('(')]
            real_paths[real_path][3] += 1

            path = ''
            for rule in info[1:-1]:
                end = rule.find('(')
                start = rule[:end].rfind(' ')
                tmp_rule = rule[start + 1:end]
                if tmp_rule.startswith('inv_'):
                    tmp_rule = invs[tmp_rule[:5]] + tmp_rule[5:]
                if len(path) == 0:
                    path += tmp_rule
                else:
                    path += '_' + tmp_rule

            print(path, real_path)

            if path == real_path and real_paths[real_path][0] == 0:
                if real_paths[real_path][3] <= 10:
                    real_paths[real_path][0] = 1
                if real_paths[real_path][3] <= 5:
                    real_paths[real_path][1] = 1
                if real_paths[real_path][3] <= 1:
                    real_paths[real_path][2] = 1

    hits10, hits5, hits1 = 0, 0, 0
    for (k, v) in real_paths.items():
        hits10 += v[0]
        hits5 += v[1]
        hits1 += v[0]

    print("Hits at top 10 is {}".format(hits10 / len(real_paths)))
    print("Hits at top 5 is {}".format(hits5 / len(real_paths)))
    print("Hits at top 1 is {}".format(hits1 / len(real_paths)))


if __name__ == '__main__':
    # integer_maze = IntegerMaze(len_path=4, num_paths=1, dir='../datasets/maze')
    eval_path()


