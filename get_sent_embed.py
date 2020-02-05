#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : get_sent_embed.py
# @Author: muyao
# @Date  : 2020/2/4/004
# 总结bert_eng和bert_chs 把句子转向量封装成函数

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


def getSentEmb(sentence, tokenizer, model):

    # 句子切分示例1
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentence = tokenizer.tokenize(marked_sentence)
    print("分词后的句子: ", tokenized_sentence)

    # 调用tokenizer来匹配tokens在词汇表中的索引
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)

    # Segment ID 区分句子： 单句id全为1；若两个句子，第一句加[SEP] 所有token赋值为0，第二句所有token赋值为1。
    segments_ids = [1] * len(tokenized_sentence)

    # 把输入list转换为PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])  # 索引list -> tensor
    segments_tensors = torch.tensor([segments_ids])  # seg嵌入list -> tensor

    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        # 这个模型的全部隐藏状态存储在对象“encoded_layers”中。这个对象有四个维度，顺序如下：
        #     1.层数(12层)
        #     2.batch号(1句)
        #     3.单词/令牌号(在我们的句子中有14个令牌)
        #     4.隐藏单元/特征号(768个特征)

        batch_i = 0
        # 按token摘出来, 每个token 12层中每层有768维度
        token_embeddings = []  # [# tokens, # layers, # features]
        for token_i in range(len(tokenized_sentence)):
            # Holds 12 layers of hidden states for each token
            hidden_layers = []
            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)

        # 下面分别用最后四层的 横向拼接/直接求和 来创建单词向量
        concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0)
                                      for layer in token_embeddings]  # [number_of_tokens, 3072]
        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0)
                                for layer in token_embeddings]  # [number_of_tokens, 768]

        # 句子向量方式1：每个token的倒数第二层表示 求平均
        sentence_embedding1 = torch.mean(encoded_layers[11], 1)

        # 句子向量方式2：[CLS] token12层表示的最后4层相加
        sentence_embedding2 = summed_last_4_layers[0].reshape(1, -1)

        # 句子向量方式3：[CLS] token12层表示的最后4层拼接
        sentence_embedding3 = concatenated_last_4_layers[0].reshape(1, -1)

        return sentence_embedding1, sentence_embedding2, sentence_embedding3


if __name__ == '__main__':
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('F:\\Corpus\\chinese_L-12_H-768_A-12')
    print("词汇表长度 =", len(tokenizer.vocab.keys()))

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(pretrained_model_name_or_path='F:\\Corpus\\chinese_L-12_H-768_A-12',
                                      from_tf=False, cache_dir='cache')

    # example sentence
    print('\n', '=' * 100)
    print("验证哪种句子编码方式效果最好 目前来看是第一种")
    sentence1 = "我和我的祖国，一刻也不能分割。"
    sentence2 = "俺和俺那国家，一会儿也分不开。"

    em11, em12, em13 = getSentEmb(sentence1, tokenizer, model)
    em21, em22, em23 = getSentEmb(sentence2, tokenizer, model)

    print("每个token的倒数第二层表示,句子相似度 =", cosine_similarity(em11, em21))
    print("[CLS] token12层表示的最后4层相加（768维）,句子相似度 =", cosine_similarity(em12, em22))
    print("[CLS] token12层表示的最后4层拼接（768*4维）,句子相似度 =", cosine_similarity(em13, em23))

    print('\n', '=' * 100)
    print("两个毫不相关的句子相似度测试")
    sentence3 = "天气不错啊今天，出去逛逛。"
    em31, em32, em33 = getSentEmb(sentence3, tokenizer, model)
    print("“祖国”有关句子和“天气”有关句子相似度1 =", cosine_similarity(em11, em31))
    print("“祖国”有关句子和“天气”有关句子相似度2 =", cosine_similarity(em12, em32))
    print("“祖国”有关句子和“天气”有关句子相似度3 =", cosine_similarity(em13, em33))
