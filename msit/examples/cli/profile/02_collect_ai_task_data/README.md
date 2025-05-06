# Collect AI Task Data
## 简介
介绍msit profile和采集AI任务运行性能数据相关的可选命令
- **命令示例**
```
msit profile msprof --application "msit benchmark -om /home/HwHiAiUser/resnet101_bs1.om" --output=output_data/ --model-execution=on --task-time=on --aicpu=on
```

## 涉及可选命令
| 参数名                    | 描述                                       | 必选   |
  | ------------------------ | ---------------------------------------- | ---- |
  | -o, --output             | 搜集到的profiling数据的存放路径，默认为当前路径下输出output目录                                                                | 否    |
  | --model-execution        | 控制ge model execution性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --task-time              | 控制ts timeline数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --aicpu                  | aicpu开关，可选on或off，默认为on。 | 否  |
  | --runtime-api            | 控制runtime api性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否 |

  ## 输出文件采集内容介绍
- **分析结果输出的目录结构**
```bash
|--- output_data/
|    |--- profiler/  # 采集的性能数据
|    |    |--- PROF_000001_20230608201922856_LPKNFOADMAQRMDGC/ # msprof保存的数据
|    |    |    |--- host/ # host侧数据
|    |    |    |--- device_0/ #device侧数据，调用多个device推理会有多个和device文件夹
|    |    |    |    |--- timeline/
|    |    |    |    |--- summary/
|    |    |    |    |--- data/
```

- **采集内容（timeline文件夹内）**

| timeline文件名 | 关联参数 | 说明 |
| ----- | ----- | ----- |
|msprof*.json|所有可生成数据的参数均会在此文件写入数据。|timeline数据总表。对采集到的timeline性能数据按照迭代粒度进行性能展示。详情请参见[timeline数据总表](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0064.html)|
|ai_stack_time_*.json|默认存在|各个组件（AscendCL，GE，Runtime，Task Scheduler）的耗时。详情参见[AscendCL、GE、Runtime、Task Schduler组件耗时数据概览](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0064.html)|
|thread_group_*.json|默认存在|AscendCL，GE，Runtime组件耗时数据。该文件内的各组件数据按照线程（Thread）粒度进行排列，方便查看各线程下各组件的耗时数据。当模型为动态Shape时自动采集并生成该文件。文件详情请参见[AscendCL、GE、Runtime组件耗时完整数据（按线程粒度展示](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0064.html)|
|task_time_*.json|--task-time|Task Scheduler任务调度信息。文件详情请参见[Task Scheduler任务调度信息数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0067.html)|
|acl_*.json|默认存在|AscendCL接口耗时数据。文件详情请参见[AscendCL接口耗时数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000111.html)|
|runtime_api_*.json|--runtime-api|Runtime接口耗时数据。文件详情请参见[Runtime接口耗时数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000111.html)|
|ge_*.json|--model-execution|GE接口耗时数据。文件详情请参见[GE接口耗时数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000111.html)|
|step_trace_*.json|默认存在|迭代轨迹数据，每轮迭代的耗时。文件详情请参见[迭代轨迹数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0069.html)|

- **采集内容（summary文件夹内）**

| summary文件名 | 关联参数 | 说明 |
| ----- | ----- | ----- |
|acl*.json|默认存在|AscendCL接口的耗时。详情请参见[AscendCL接口耗时数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000111.html)|
|acl_statistic_*.csv|配置--task-time生成AI Core算子信息；配置--aicpu生成AI CPU算子信息。 |AI Core和AI CPU算子信息。详情请参见[AI Core和AI CPU算子数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0072.html)|
|op_statistic_*.csv|配置--task-time生成AI Core算子信息；配置--aicpu生成AI CPU算子信息。|AI Core和AI CPU算子调用次数及耗时，从算子类型维度找出耗时最大的算子类型。详情请参见[AI Core和AI CPU算子调用次数及耗时数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0073.html)|
|step_trace_*.csv|默认存在|迭代轨迹数据。文件详情请参见[迭代轨迹数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0069.html)|
|ai_stack_time_*.csv|默认存在|每个组件（AscendCL、GE、Runtime、Task Scheduler）的耗时。详情请参见[各个组件的耗时数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0068.html)|
|runtime_api_*.csv|--runtime-api|每个runtime api的调用时长。详情请参见[Runtime接口耗时数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000111.html)|
|fusion_op_*.csv|默认存在|模型中算子融合前后信息。详情请参见[模型中算子融合前后信息数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0078.html)|
|task_time_*.csv|--task-time|Task Scheduler的任务调度信息数据。详情请参见：[Task Scheduler的任务调度信息数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0067.html)|
|aicpu_*.csv|--aicpu|AI CPU数据。文件详情请参见[AI CPU数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0076.html)|
|prof_rule_0.json|默认存在|调优建议。无需指定Profiling参数自动生成，完成后打屏显示结果，详细介绍请参见[性能调优建议](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0065.html)|