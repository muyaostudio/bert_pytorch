#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tf2torch.py
# @Author: muyao
# @Date  : 2020/2/4/004

# 把TensorFlow的ckpt转成torch bin
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    # convert_tf_checkpoint_to_pytorch(tf_checkpoint_path="F:\\Corpus\\uncased_L-12_H-768_A-12\\bert_model.ckpt",
    #                                  bert_config_file="F:\\Corpus\\uncased_L-12_H-768_A-12\\bert_config.json",
    #                                  pytorch_dump_path="F:\\Corpus\\uncased_L-12_H-768_A-12\\bert_model.bin")
    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path="F:\\Corpus\\chinese_L-12_H-768_A-12\\bert_model.ckpt",
                                     bert_config_file="F:\\Corpus\\chinese_L-12_H-768_A-12\\bert_config.json",
                                     pytorch_dump_path="F:\\Corpus\\chinese_L-12_H-768_A-12\\pytorch_model.bin")
