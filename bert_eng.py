#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: muyao
# @Date  : 2020/2/4/004

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
# F:\Corpus\uncased_L-12_H-768_A-12

tokenizer = BertTokenizer.from_pretrained('F:\\Corpus\\uncased_L-12_H-768_A-12')
print("tokenizer: ", tokenizer, '\n')

# BERT tokenizer模型的词汇量限制大小为30,000
# 句子切分示例1
text1 = "Here is the sentence I want embeddings for the sentence."
marked_text1 = "[CLS] " + text1 + " [SEP]"
print("marked_text1: ", marked_text1)
tokenized_text1 = tokenizer.tokenize(marked_text1)
print("tokenized_text1: ", tokenized_text1, '\n')

# 句子切分示例2
text2 = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
marked_text2 = "[CLS] " + text2 + " [SEP]"
print("marked_text2: ", marked_text2)
tokenized_text2 = tokenizer.tokenize(marked_text2)
print("tokenized_text2: ", tokenized_text2, '\n')

# 下面是词汇表中包含的一些token示例。以两个#号开头的标记是子单词或单个字符。
print("词汇表中包含的一些token示例: ", list(tokenizer.vocab.keys())[5000:5020])
print("词汇表长度 =", len(tokenizer.vocab.keys()), '\n')

# 调用tokenizer来匹配tokens在词汇表中的索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text1)
print("tokenized_text1: ", tokenized_text1)
print("tokenized_text1中每个词在词汇表的索引(indexed_tokens): ", indexed_tokens)
for tup in zip(tokenized_text1, indexed_tokens):
    print(tup)
print('\n')

# Segment ID 区分句子： 单句id全为1；若两个句子，第一句加[SEP] 所有token赋值为0，第二句所有token赋值为1。
segments_ids = [1] * len(tokenized_text1)
print("segments_ids: ", segments_ids, '\n')

# 开始调用bert
# 把输入list转换为PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])  # 索引list -> tensor
segments_tensors = torch.tensor([segments_ids])  # seg嵌入list -> tensor
print("tokens_tensor: ", tokens_tensor, tokens_tensor.shape)
print("segments_tensors: ", segments_tensors, segments_tensors.shape, '\n')

# Load pre-trained model (weights)
print("=========load model=========")
model = BertModel.from_pretrained(pretrained_model_name_or_path='F:\\Corpus\\uncased_L-12_H-768_A-12',
                                  from_tf=False, cache_dir='cache')
# 模型置于评估模式，而不是训练模式 关闭了训练中使用的dropout正则化
model.eval()
# print(model, "\n")

# Predict hidden states features for each layer
# torch.no_grad禁用梯度计算，节省内存，并加快计算速度(我们不需要梯度或反向传播，因为我们只是运行向前传播)。
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # 这个模型的全部隐藏状态存储在对象“encoded_layers”中。这个对象有四个维度，顺序如下：
    #     1.层数(12层)
    #     2.batch号(1句)
    #     3.单词/令牌号(在我们的句子中有14个令牌)
    #     4.隐藏单元/特征号(768个特征)
    print("len of encoded_layers(type:list): ", len(encoded_layers))
    print("encoded_layers[0].shape: ", encoded_layers[0].shape)

    print("Number of layers:", len(encoded_layers))
    layer_i = 0
    print("Number of batches:", len(encoded_layers[layer_i]))
    batch_i = 0
    print("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    token_i = 0
    print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]), '\n')

    # 查看一下给定层5和token5的值范围
    token_i = 5
    layer_i = 5
    vec = encoded_layers[layer_i][batch_i][token_i]
    # 绘制直方图
    plt.figure(figsize=(10, 10))
    plt.hist(vec, bins=200)
    plt.show()

    # 按token摘出来, 每个token 12层中每层有768维度
    token_embeddings = []  # [# tokens, # layers, # features]
    for token_i in range(len(tokenized_text1)):
        # Holds 12 layers of hidden states for each token
        hidden_layers = []
        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]
            hidden_layers.append(vec)
        token_embeddings.append(hidden_layers)
    # Sanity check the dimensions:
    print("Number of tokens in sequence:", len(token_embeddings))
    print("Number of layers per token:", len(token_embeddings[0]))

    # 每个token有12个768dim的表示 怎么得到token的最终表示呢？
    # bert作者实验了一波发现 最后四层横向拼接(768 768 768 768) 效果最好

    # 下面分别用最后四层的 横向拼接/直接求和 来创建单词向量
    concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0)
                                  for layer in token_embeddings]  # [number_of_tokens, 3072]
    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0)
                            for layer in token_embeddings]  # [number_of_tokens, 768]

    # 句子向量：每个token的倒数第二层表示 求平均
    sentence_embedding = torch.mean(encoded_layers[11], 1)
    print("句子向量shape:", sentence_embedding[0].shape, '\n')

    # 验证上下文相关性 ：输出每个token的768维embed的前5维,看看句子中相同单词“sentence”编码是不是一样（不一样）
    for idx, token in enumerate(tokenized_text1):
        print(idx, token,
              '\n最后四层相加的方式(前5维)：', summed_last_4_layers[idx][:5],
              '\n最后四层拼接的方式(前5维)：', concatenated_last_4_layers[idx][:5])

    print(cosine_similarity(summed_last_4_layers[4].reshape(1, -1),
                            summed_last_4_layers[13].reshape(1, -1))[0][0])
    print(cosine_similarity(concatenated_last_4_layers[4].reshape(1, -1),
                            concatenated_last_4_layers[13].reshape(1, -1))[0][0])
