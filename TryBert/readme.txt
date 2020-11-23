train.txt  BERT训练命令
predict.txt  BERT预测命令 
train_data_sample.json 训练数据样例
test_data_sample.json 测试数据样例

在Bert上采用自己的阅读理解数据集进行训练、预测需要修改run_squad.py中的函数read_squad_examples()，以便生成模型需要的数据样例。
