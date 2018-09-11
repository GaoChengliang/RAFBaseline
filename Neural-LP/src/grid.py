from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import random
import numpy as np


class Grid(object):

    def __init__(self, border_length, path_length, dir, random_seed=1111):
        self.border_length = border_length
        self.path_length = path_length
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.directions = {
            'North': (0, 1),
            'South': (0, -1),
            'East': (1, 0),
            'West': (-1, 0),
            'NorthEast': (1, 1),
            'NorthWest': (-1, 1),
            'SouthEast': (1, -1),
            'SouthWest': (-1, -1)
        }

        random.seed(random_seed)
        np.random.seed(random_seed)
        self._generate_path()
        self.write_entities()
        self.write_relations()
        self.write_facts()
        self.write_data()

    def _generate_path(self):
        self.pre_define_paths = [
            ['East', 'East', 'East', 'East', 'East', 'East'],
            ['South', 'South', 'South', 'South', 'East', 'East'],
            ['South', 'South', 'SouthWest', 'SouthWest', 'East', 'East'],
            ['East', 'East', 'South', 'South', 'West', 'West', 'North', 'North'],
        ]
        self.path = np.random.choice(self.directions.keys(), self.path_length)
        self.path_str = '_'.join(self.path)

    def _get_dst(self, x, y):
        for i in range(self.path_length):
            dirs = self.directions[self.path[i]]
            x, y = x + dirs[0], y + dirs[1]
            if x < 1 or x > self.border_length or y < 1 or y > self.border_length:
                return -1, -1
        return x, y

    def write_data(self):
        train_path = self.dir + '/train.txt'
        test_path = self.dir + '/test.txt'

        data = []
        for i in range(1, self.border_length + 1):
            for j in range(1, self.border_length + 1):
                dst_x, dst_y = self._get_dst(i, j)
                if dst_x == -1 and dst_y == -1:
                    data.append('({}_{})\t{}\tOOB\n'.format(i, j, self.path_str))
                else:
                    data.append('({}_{})\t{}\t({}_{})\n'.format(i, j, self.path_str, dst_x, dst_y))

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
            for i in range(1, self.border_length + 1):
                for j in range(1, self.border_length + 1):
                    f.write('({}_{})\n'.format(i, j))
            f.write('OOB\n')

    def write_relations(self):
        path = self.dir + '/relations.txt'
        with open(path, 'w') as f:
            f.write('North\n')
            f.write('South\n')
            f.write('East\n')
            f.write('West\n')
            f.write('NorthEast\n')
            f.write('NorthWest\n')
            f.write('SouthEast\n')
            f.write('SouthWest\n')
            f.write(self.path_str + '\n')

    def write_facts(self):
        path = self.dir + '/facts.txt'
        with open(path, 'w') as f:
            for i in range(1, self.border_length + 1):
                for j in range(1, self.border_length + 1):
                    for (k, v) in self.directions.items():
                        x, y = i + v[0], j + v[1]
                        if x < 1 or x > self.border_length or y < 1 or y > self.border_length:
                            f.write('({}_{})\t{}\tOOB\n'.format(i, j, k))
                        else:
                            f.write('({}_{})\t{}\t({}_{})\n'.format(i, j, k, x, y))


def eval_path():
    hits10, hits5, hits1 = 0, 0, 0

    invs = {
        'inv_North': 'South',
        'inv_South': 'North',
        'inv_West': 'East',
        'inv_East': 'West',
        'inv_NorthEast': 'SouthWest',
        'inv_NorthWest': 'SouthEast',
        'inv_SouthEast': 'NorthWest',
        'inv_SouthWest': 'NorthEast'
    }

    rules = '../exps/grid/rules.txt'
    with open('../datasets/grid/relations.txt', 'r') as path_file:
        real_path = path_file.readlines()[-1].strip()

    with open(rules, 'r') as rules_file:
        line_num = 0
        for line in rules_file:
            if line.strip().split('\t')[1].startswith('inv_'):
                continue

            line_num += 1
            info = line.strip().split('\t')[1].split(')')
            path = ''
            for rule in info[1:-1]:
                end = rule.find('(')
                start = rule[:end].rfind(' ')
                tmp_rule = rule[start + 1:end]
                if tmp_rule.startswith('inv_'):
                    tmp_rule = invs[tmp_rule]
                if len(path) == 0:
                    path += tmp_rule
                else:
                    path += '_' + tmp_rule

            print(line_num, path, real_path)

            if path == real_path:
                if line_num <= 10:
                    hits10 = 1
                if line_num <= 5:
                    hits5 = 1
                if line_num <= 1:
                    hits1 = 1
                break

    print("Hits at top 10 is {}".format(hits10))
    print("Hits at top 5 is {}".format(hits5))
    print("Hits at top 1 is {}".format(hits1))


if __name__ == '__main__':
    # grid = Grid(border_length=16, path_length=4, dir='../datasets/grid', random_seed=1322)
    eval_path()
