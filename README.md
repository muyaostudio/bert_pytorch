# bert_pytorch
pytorch-pretrained-BERT实战，包括英文和中文版，并集成了句子to嵌入向量的函数

Step 1
下载项目至本地

Step 2
根目录下新建cache文件

Step 3
运行tf2torch.py => 把tf格式的model转成torch能识别的格式(.bin)
运行bert_eng.py => 英文bert
运行bert_chs.py => 中文bert
运行get_sent_embed.py => 句子转embedding表示
