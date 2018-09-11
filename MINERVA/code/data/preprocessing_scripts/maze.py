from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import csv

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

        self.write_facts()
        self.write_data()
        self.write_vocab()

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

    def write_data(self):
        train_path = self.dir + '/train.txt'
        eval_path = self.dir + '/dev.txt'
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
        train_data, eval_data, test_data = data[:int(len(data) * 1 / 2)], data[int(len(data) * 1 / 2): int(
            len(data) * 3 / 4)], data[int(len(data) * 3 / 4):]

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
        path = self.dir + '/graph.txt'
        with open(path, 'w') as f:
            for i in range(self.min_integer, self.max_integer + 1):
                for ops in self.ops_pool:
                    for op in ops:
                        f.write('{}\t{}\t{}\n'.format(i, op(0)[0], op(i)[1]))

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
    hits10, hits5, hits1 = 0.0, 0.0, 0.0

    test_path = '../../../datasets/data_preprocessed/maze/test.txt'
    out_path = '../../../output/maze/528c_3_0.07_100_0.0/test_beam'
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
    integer_maze = IntegerMaze(len_path=4, num_paths=1, dir='../../../datasets/data_preprocessed/maze')
