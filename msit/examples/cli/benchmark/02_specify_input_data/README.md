# Specify Input Data

## 介绍

默认情况下，构造全为0的数据送入模型推理。可指定文件输入或者文件夹输入。

## 运行示例
1. 文件输入场景。

    使用--input参数指定模型输入文件，多个文件之间通过“,”进行分隔。

    本场景会根据文件输入size和模型实际输入size进行对比，若缺少数据则会自动构造数据补全，称为组Batch。

    ```bash
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./1.bin,./2.bin,./3.bin,./4.bin,./5.bin
    ```

2. 文件夹输入场景。

    使用input参数指定模型输入文件所在目录，多个目录之间通过“,”进行分隔。

    本场景会根据文件输入size和模型实际输入size进行组Batch。

    ```bash
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./
    ```

    模型输入需要与传入文件夹的个数一致。

    例如，bert模型有三个输入，则必须传入3个文件夹，且三个文件夹分别对应模型的三个输入，顺序要对应。
    模型输入参数的信息可以通过开启调试模式查看，bert模型的三个输入依次为input_ids、 input_mask、 segment_ids，所以依次传入三个文件夹：

    - 第一个文件夹“./data/SQuAD1.1/input_ids"，对应模型第一个参数"input_ids"的输入
    - 第二个文件夹"./data/SQuAD1.1/input_mask"，对应第二个输入"input_mask"的输入
    - 第三个文件夹"./data/SQuAD1.1/segment_ids"，对应第三个输入"segment_ids"的输入

    ```bash
    msit benchmark --om-model ./save/model/BERT_Base_SQuAD_BatchSize_1.om --input ./data/SQuAD1.1/input_ids,./data/SQuAD1.1/input_mask,./data/SQuAD1.1/segment_ids
    ```

## FAQ
使用出现问题时，可参考[FAQ](https://gitee.com/ascend/msit/wikis/benchmark_FAQ/msit%20benchmark%20%E5%AE%89%E8%A3%85%E9%97%AE%E9%A2%98FAQ)
