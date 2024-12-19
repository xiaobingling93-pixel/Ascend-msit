# msit graph inspect 使用介绍

## 简介

为了提高训推性能，需要将 GE 图中的动态 shape 算子替换，得到全静态图。因此，首先需要筛选图中的动态 shape 算子。

## 安装

源码安装 components：

```sh
git clone https://gitee.com/ascend/msit.git
cd msit/msit

pip install .
```
安装 graph 组件：

```sh
cd ./components/graph

python setup.py bdist_wheel
cd ./dist
pip install msit*.whl
```

## 图扫描命令行入口

```sh
msit graph inspect <options>
```

参数说明：

|参数名|是否必选|使用说明|
|-----|-----|-----|
|--graph-path, -gp|是| pbtxt 文件的路径。|
|--type, -t|是|指定具体的扫描类别，目前仅支持“动态 shape”（dshape）。|
|--log-level, -l|否|日志等级，debug、info、warning、error|
|--output, -o|否|输出目录，当前输出表头为 Op_name, Input_name, and Output_name，默认是 “./”。|

命令示例：

```sh
msit graph inspect -gp ./test_pbgraph.pbtxt -t dshape -o ./output
```
