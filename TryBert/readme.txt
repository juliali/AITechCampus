文件：
======
script.txt  BERT训练命令、预测命令 
train_data_sample.json 训练数据样例
test_data_sample.json 测试数据样例

运行环境：
==========
tensorflow 1.11.0
python 3.6

BERT 代码及预训练模型：
======================
BERT下载地址：https://github.com/google-research/bert
BERT下载方式：（1）直接下载，或者（2）git clone BERT github repo

中文预训练模型chinese_L-12_H-768_A-12下载地址：
https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

将BERT代码解压缩或git pull至 ${BERT_ROOT} 目录，然后创建 ${BERT_ROOT}/model 和 ${BERT_ROOT}/data 子目录，将 chinese_L-12_H-768_A-12.zip 拷贝到 ${BERT_ROOT}/model，并就地解压缩。

在${BERT_ROOT}/data 目录中放置 train.json和test.json文件，可以直接将 train_data_sample.json 拷贝到 ${BERT_ROOT}/data 并更名为 train.json, test.json同理。


说明：
1. 训练和预测命令可在windows环境下执行。
2. 在视频中我们采用了官方的代码和相对应数据格式的数据集进行训练、预测。如果采用自己的阅读理解数据集，需要修改run_squad.py中的函数read_squad_examples()，以便生成模型需要的数据样例。
3. 用官方代码predict写入文件时，会将中文写为unicode编码，可以在 http://www.jsons.cn/unicode/ 上进行转换。