# 解析推理服务化数据为trace格式

该工具支持将从MindIE Server采集的性能数据解析为trace格式。

## 1 数据准备

准备已经使用msprof进行初步解析的性能数据，确认存在数据文件`${your_file}/PROF_XXXX/host/sqlite/msproftx.db`，并且数据库中存在名为`MsprofTxEx`的表。


## 2 执行解析脚本
进入`msserverprofiler`文件夹，执行`parse_data_to_trace.py`脚本，具体命令如下：
```commandline
python3 parse_data_to_trace.py --input ${your_file} --output ${output_dir}
```
### 命令行参数

| 参数名                           | 描述                                        | 必选 |
|-------------------------------|-------------------------------------------| ---- |
| --input                       | 指定数据所在文件夹，会遍历读取该文件夹下所有名为`msproftx.db`的数据库 | 是   |
| --output                      | 指定解析后json文件生成路径，不传参默认在当前文件夹下生成            | 否   |

运行成功后，会在`${output_dir}`下生成`chrome_tracing_${timestamp}.json`文件

## 3 可视化
打开`chrome://tracing`网页，点击`load`将`chrome_tracing_${timestamp}.json`上传，即可查看可视化结果。