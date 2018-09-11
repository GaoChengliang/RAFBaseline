from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import csv

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
        # self.write_entities()
        # self.write_relations()
        self.write_facts()
        self.write_data()
        self.write_vocab()

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
        eval_path = self.dir + '/dev.txt'
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
        train_data, eval_data, test_data = data[:int(len(data) * 1 / 2)], data[int(len(data) * 1 / 2): int(len(data) * 3 / 4)], data[int(len(data) * 3 / 4):]

        with open(train_path, 'w') as f:
            for line in train_data:
                f.write(line)

        with open(eval_path, 'w') as f:
            for line in eval_data:
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
        path = self.dir + '/graph.txt'
        with open(path, 'w') as f:
            for i in range(1, self.border_length + 1):
                for j in range(1, self.border_length + 1):
                    for (k, v) in self.directions.items():
                        x, y = i + v[0], j + v[1]
                        if x < 1 or x > self.border_length or y < 1 or y > self.border_length:
                            f.write('({}_{})\t{}\tOOB\n'.format(i, j, k))
                        else:
                            f.write('({}_{})\t{}\t({}_{})\n'.format(i, j, k, x, y))

    def write_vocab(self):
        vocab_dir = self.dir + '/vocab'
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)

        entity_vocab = {}
        relation_vocab = {}

        entity_vocab['PAD'] = len(entity_vocab)
        entity_vocab['UNK'] = len(entity_vocab)
        relation_vocab['PAD'] = len(relation_vocab)
        relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
        relation_vocab['NO_OP'] = len(relation_vocab)
        relation_vocab['UNK'] = len(relation_vocab)

        entity_counter = len(entity_vocab)
        relation_counter = len(relation_vocab)

        for f in ['/train.txt', '/dev.txt', '/test.txt', '/graph.txt']:
            with open(self.dir + f) as raw_file:
                csv_file = csv.reader(raw_file, delimiter='\t')
                for line in csv_file:
                    e1, r, e2 = line
                    if e1 not in entity_vocab:
                        entity_vocab[e1] = entity_counter
                        entity_counter += 1
                    if e2 not in entity_vocab:
                        entity_vocab[e2] = entity_counter
                        entity_counter += 1
                    if r not in relation_vocab:
                        relation_vocab[r] = relation_counter
                        relation_counter += 1

        with open(vocab_dir + '/entity_vocab.json', 'w') as fout:
            json.dump(entity_vocab, fout)

        with open(vocab_dir + '/relation_vocab.json', 'w') as fout:
            json.dump(relation_vocab, fout)


def eval_path():
    hits10, hits5, hits1 = 0, 0, 0

    test_path = '../../../datasets/data_preprocessed/grid/test.txt'
    out_path = '../../../output/grid/528c_3_0.07_100_0.0/test_beam'
    pre_paths = {}

    file_list = os.listdir(out_path)
    for file_name in file_list:
        if file_name.startswith('paths_'):
            with open(out_path + '/' + file_name, 'r') as out_file:
                path_name = file_name[file_name.find('_') + 1:]
                pre_paths[path_name] = {}
                while True:
                    line = out_file.readline()
                    if line is None or len(line) == 0:
                        break
                    s, _ = line.strip().split('\t')[:]
                    pre_paths[path_name][s] = []
                    while True:
                        path_str = out_file.readline()
                        if path_str.startswith('#####'):
                            break
                        path = path_str.strip().split('\t')
                        path_without_no_op = ''
                        for ss in path:
                            if ss != 'NO_OP':
                                path_without_no_op += ss + '_'
                        path_without_no_op = path_without_no_op[:-1]
                        pre_paths[path_name][s].append(path_without_no_op)

    cnt = 0
    with open(test_path, 'r') as test_file:
        for line in test_file:
            cnt += 1
            s, r, _ = line.strip().split('\t')
            if r in pre_paths[r][s]:
                i = pre_paths[r][s].index(r)
                if i < 10:
                    hits10 += 1
                if i < 5:
                    hits5 += 1
                if i < 1:
                    hits1 += 1

    print("Path hits at top 10 is {}".format(hits10 / cnt))
    print("Path hits at top 5 is {}".format(hits5 / cnt))
    print("Path hits at top 1 is {}".format(hits1 / cnt))


if __name__ == '__main__':
    # grid = Grid(border_length=64, path_length=2, dir='../../../datasets/data_preprocessed/grid', random_seed=666)
    eval_path()
