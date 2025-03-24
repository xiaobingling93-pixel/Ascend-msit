# vllm_profiler工具使用说明

vllm是当前推理服务化主流框架，华为云Ascend vllm框架支持昇腾生态。本工具将基于华为云Ascend vllm，支持profiling数据采集能力，便于进行服务化推理调试调优。

## 环境准备
1. 正常可以运行ascend_vllm推理服务
2. 获取msit仓源码
3. 正常安装cann包并source对应环境变量

## 操作步骤
1. 修改 vllm 源码，在init.py 入口处添加 vllm_profiler相关导入。
命令行输入`pip show vllm`，返回vllm安装路径`${vllm_install_path}`，在`${vllm_install_path}/vllm/__init__.py`文件中添加
```
import vllm_profiler.vllm_profiler_0_6_3
```
2. 配置 msserverprofiler 路径，启动 vllm server。
将msit源码中`msserviceprofiler/ms_service_profiler_ext`路径加入PYTHONPATH
3. 设置profiler打点环境变量及配置文件。
服务化性能数据采集通过json配置文件，配置采集开关、保存路径等。以ms_service_profiler_config.json文件名为例，例如设置数据落盘路径为`${logs_prof}`，
```
export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json
```
其中ms_service_profiler_config.json中包含文件内容：
```
{
    "enable": 0,
    "prof_dir": "${logs_prof}",
    "profiler_level": "L1"
}
```
参数说明：
|参数|说明|是否必选|
| ---- | ---- | ---- |
|enable|是否开启性能数据采集的开关，取值为：0，关闭；1，开启|是|
|prof_dir|采集到的性能数据的存放路径。默认值为$HOME/.ms_server_profiler|否|
|profiler_level|数据采集等级，取值为：L0，异常级别的性能数据；L1，普通级别的性能数据，默认值；L2，详细级别的性能数据；L3，冗长的性能数据。|否|

**注意**：拉起vllm框架前，就需要配置SERVICE_PROF_CONFIG_PATH环境变量，其中ms_service_profiler_config.json文件中的enable字段必须设置为0；当vllm框架拉起成功后，再将enable字段修改为1，再发送请求即可落盘profiling数据。

4. 可选：指定卡运行命令，例如`export ASCEND_RT_VISIBLE_DEVICES=1`为指定1卡运行
5. 拉取vllm框架，发送请求，在步骤3中的ms_service_profiler_config.json设置的`${logs_prof}`路径下，会落盘profiling数据
6. 调用`msprof --export=on --output=${logs_prof}/PROF_xxx_xxx_xxx`命令行处理`${logs_prof}`目录下所有的落盘数据，生成msproftx.db文件

# 结果说明
### 1. 执行推理处理时间数据
① modelExec，表示模型执行时间
```
rid: 请求ID
batch_type: batch类型
batch_size: batch大小
```

② Forward, 模型前向计算时间
```
rid: 请求ID
```

### 2. 请求队列管理状态变化，及组batch过程数据
① Enqueue, Dequeue，表示请求入队、出队
```
rid: 请求ID
QueueSize：当前队列大小
scope：队列名称，通常含有waiting、running队列
```

② BatchSchedule，表示调度信息
```
rid: 当前调度batch中的请求ID列表
QueueSize：当前队列大小
iter_size:当前迭代返回token长度
```

③ ReqState，表示请求状态变化信息
```
rid: 请求ID
WAITING+、RUNNING+、FINIEHED+：请求状态名称，数字-1表示当前状态请求数-1，1则为当前请求状态数+1
```

### 3. kv cache数据
① Allocate，表示请求分配kvcache block字段
```
domain：表示当前为kvcache相关信息
rid: 请求ID
deviceBlock: 分配的block数量
```

② AppendSlot，表示请求过程中新增内存进行缓存的字段
```
domain：表示当前为kvcache相关信息
rid: 请求ID
deviceBlock: 追加的block数量
```

③ Free，请求过程中释放的缓存字段
```
domain：表示当前为kvcache相关信息
rid: 请求ID
deviceBlock: 释放的block数量
```

④ GetCacheHitRate，请求过程中缓存的命中率
```
domain：表示当前为kvcache相关信息
cpuHitCache: cpu缓存命中率
hitCache: gpu缓存命中率
```

### 4. request数据
① httpReq，表示请求到达
```
domain：表示当前为http请求相关信息
rid: 请求ID
```

② recvTokenSize，表示请求输入长度
```
domain：表示当前为http请求相关信息
rid: 请求ID
```

③ replyTokenSize，表示请求输出长度
```
domain：表示当前为http请求相关信息
rid: 请求ID
```