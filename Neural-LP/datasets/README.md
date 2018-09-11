这个model只支持把关系输入进去，最后推断关系由哪些knowledge推断而来

## 数据生成
### grid.py 和 maze.py
+ 每次生成一个数据集，需要指定存储路径
+ 会生成5个文件，分别为
> 1. entities.txt 记录所有的实体（数字，坐标等）
> 2. relations.txt 记录所有的一步转换关系，和生成的path字符串
> 3. facts.txt 由一步转换关系生成的数据，一些knowledge
> 4. train.txt 为训练集， test.txt 为测试集。两个文件的data数量是3：1。格式为entity path entity

## 代码运行
### 训练
```
python src/main.py --datadir=datasets/grid --exps_dir=exps/ --exp_name=grid --num_step=5 --max_epoch=100 --min_epoch=50 --batch_size=16 --top_k=5
```

以grid为例, num_step参数比较重要表示最多路径走几步 + 1， 指定好max_epoch和min_epoch，代码会在最优结果停机，使用最优结果。
训练完后，会在exps_dir/exp_name文件夹下产生所有学习到的规则

### 测试
```
. eval/collect_all_facts.sh datasets/grid
python eval/get_truths.py datasets/grid
python eval/evaluate.py --preds=exps/grid/test_predictions.txt --truths=datasets/grid/truths.pckl
```
在测试集上计算dst的指标。

运行grid.py或者maze.py的eval_path函数来计算path的指标。由于path是在整个训练集上计算出来的，一旦训练结束后，规则就定下来了。

