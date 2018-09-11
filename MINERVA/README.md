# MINERVA
Meandering In Networks of Entities to Reach Verisimilar Answers 

Code and models for the paper [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851)

MINERVA is a RL agent which answers queries in a knowledge graph of entities and relations. Starting from an entity node, MINERVA learns to navigate the graph conditioned on the input query till it reaches the answer entity. For example, give the query, (Colin Kaepernick, PLAYERHOMESTADIUM, ?), MINERVA takes the path in the knowledge graph below as highlighted. Note: Only the solid edges are observed in the graph, the dashed edges are unobsrved.


## Requirements
To install the various python dependences (including tensorflow)
```
pip install -r requirements.txt
```


## 数据生成
### grid.py 和 maze.py
+ 每次生成一个数据集，需要指定存储路径
+ 会生成6个文件，分别为
> 1. graph.txt 由一步转换关系生成的数据，一些knowledge
> 2. train.txt 为训练集， dev.txt 为验证集, test.txt 为测试集。格式为entity path entity
> 3. vocab文件夹下为两个json文件，记录entity和relation的映射


## Training
Training MINERVA is easy!. The hyperparam configs for each experiments are in the [configs](https://github.com/shehzaadzd/MINERVA/tree/master/configs) directory. To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the ${dataset}.sh is the name of the config file. For example, 
```
sh run.sh configs/countries_s3.sh
```

训练结束后会运行test数据集，输出hit率

## Output
The code outputs the evaluation of MINERVA on the datasets provided. The metrics used for evaluation are Hits@{1,3,5,10,20} and MRR (which in the case of Countries is AUC-PR). Along with this, the code also outputs the answers MINERVA reached in a file.

paths_*.txt 文件记录每个测试数据的生成路径，按照概率从大到小排列
使用grid.py和maze.py的eval_path方法记录path的hit率

## Code Structure

The structure of the code is as follows
```
Code
├── Model
│    ├── Trainer
│    ├── Agent
│    ├── Environment
│    └── Baseline
├── Data
│    ├── Grapher
│    ├── Batcher
│    └── Data Preprocessing scripts
│            ├── create_vocab
│            ├── create_graph
│            ├── Trainer
│            └── Baseline

```

## Data Format

To run MINERVA on a custom graph based dataset, you would need the graph and the queries as triples in the form of (e<sub>1</sub>,r, e<sub>2</sub>).
Where e<sub>1</sub>, and e<sub>2</sub> are _nodes_ connected by the _edge_ r.
The vocab can of the dataset can be created using the create_vocab.py file found in data/preprocessng scripts. The vocab needs to be stores in the json format `{'entity/relation': ID}`.
The following shows the directory structure of the Kinship dataset.

```
kinship
    ├── graph.txt
    ├── train.txt
    ├── dev.txt
    ├── test.txt
    └── Vocab
            ├── entity_vocab.json
            └── relation_vocab.json
``` 
## Citation
If you use this code, please cite our paper
```
@inproceedings{minerva,
  title = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author = {Das, Rajarshi and Dhuliawala Shehzaad and Zaheer Manzil and Vilnis Luke and Durugkar Ishan and Krishnamurthy Akshay and Smola Alex and McCallum Andrew},
  booktitle = {ICLR},
  year = 2018
}
```
