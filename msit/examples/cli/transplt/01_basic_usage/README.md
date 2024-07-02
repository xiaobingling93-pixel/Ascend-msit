# Basic Usage

## 介绍

Transplt迁移分析工具，提供NV C++推理应用工程迁移分析以及昇腾API推荐。它使用clang等工具分析应用工程源文件中所调用到的NV加速库API、结构体以及枚举信息，并判断在昇腾库上是否有对应的API、结构体以及枚举，然后给出详细分析报告。帮助用户快速将NV C++推理应用工程迁移到昇腾设备上。

## 使用方法

```shell
ait transplt [OPTIONS]
```

OPTIONS参数说明如下

| 参数        | 说明                                                          | 是否必选 |
| ----------- | ------------------------------------------------------------ | -------- |
| -s, --source | 待迁移分析工程的目录 | 是 |
| -f, --report-type | 输出报告类型，目前支持csv(xlsx)和json两种 | 否 |
| --log-level | 日志打印级别，默认为INFO。可选项为：DEBUG，INFO，WARNING，ERROR | 否 |
| --tools | 构建工具类型，默认为cmake，目前只支持cmake类型 | 否 |
| --mode | 源码扫描模式，支持all（默认），api-only; 目前仅c++源码支持api-only | 否 |
| --help | 显示帮助信息 | 否 |

## 运行示例

```shell
ait transplt -s /workspace/sample
```

```shell
2023-06-06 16:19:46,881 - INFO - scan_api.py[123] - Scan source files...
2023-06-06 16:19:46,882 - INFO - clang_parser.py[355] - Scanning file: /workspace/sample/xxxx.cpp
2023-06-06 16:20:12,004 - INFO - cxx_scanner.py[33] - Total time for scanning cxx files is 25.12237787246704s
2023-06-06 16:20:17,799 - INFO - csv_report.py[46] - Report generated at: /workspace/sample/output.xlsx
2023-06-06 16:20:17,799 - INFO - scan_api.py[113] - **** Project analysis finished <<<
```

如果仅需要扫描C++应用工程的api，开启调用序列模板匹配功能，则需要使用如下命令：

``` shell
ait transplt -s /data/examples/simple --mode api-only
```

all和api-only两种源码扫描模式下,输出结果都保存在output.xlsx文件中。

1. all模式下(源码扫描模式:--mode all)，会记录代码中所有用到的NV加速库API、结构体以及枚举信息和昇腾的支持情况，结果如下：

| AccAPI              | CUDAEnable | Location        | Context(形参 \| 实参 \| 来源代码 \| 来源位置) | AccLib | AscendAPI                | Description                                            | Workload(人/天) | Params(Ascend:Acc) | AccAPILink | AscendAPILink                                                | AscendLib |
| ------------------- | ---------- | --------------- | --------------------------------------------- | ------ | ------------------------ | ------------------------------------------------------ | --------------- | ------------------ | ---------- | ------------------------------------------------------------ | --------- |
| CUVIDDECODECAPS     | TRUE       | xxx.cpp, 203:21 | []                                            | Codec  | hi_vdec_chn_attr         | 定义解码通道属性结构体。                               | 0.2             |                    |            |                                                              |           |
| cuvidGetDecoderCaps | TRUE       | xxx.cpp, 211:5  | [xxx]                                         | Codec  | hi_mpi_vdec_get_chn_attr | 获取视频解码通道属性。                                 | 0.1             |                    |            | https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_03_0403.html |           |
| cuvidCreateDecoder  | TRUE       | xxx.cpp, 362:5  | [xxx]                                         | Codec  | hi_mpi_vdec_create_chn   | 根据设置的通道属性创建解码通道。                       | 0.2             |                    |            | https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_03_0401.html |           |
| CUVIDPICPARAMS      | TRUE       | xxx.cpp, 526:36 | [xxx]                                         | Codec  | hi_vdec_pic_info         | 定义视频原始图像帧结构。                               | 0.2             |                    |            |                                                              |           |
| cuvidDecodePicture  | TRUE       | xxx.cpp, 534:5  | [xxx]                                         | Codec  | hi_mpi_vdec_send_stream  | 解码前，向解码通道发送码流数据及存放解码结果的buffer。 | 0.2             |                    |            |                                                              |           |
| cuCtxPopCurrent     | TRUE       | xxx.cpp, 544:5  | ['CUcontext * pctx \| \| NO_REF \| NO_REF']   | CUDA   |                          |                                                        | 0.1             |                    |            |                                                              |           |

输出数据说明：

| 标题                                          | 说明      |
| -------------- | ---------------------------------------- |
| AccAPI                                        | 三方加速库API |
| CUDAEnable                                    | 是否CUDA |
| Location                                      | 调用三方加速库API的位置 |
| Context(形参 \| 实参 \| 来源代码 \| 来源位置) | 三方加速库API参数及上下文，包括形参、实参、来源代码文件以及来源位置 |
| AccLib                                        | API所属三方加速库 |
| AscendAPI                                     | 推荐的昇腾API |
| Description                                   | API描述 |
| Workload(人/天)                               | 迁移工作量（人/天） |
| Params(Ascend:Acc) | 昇腾API和三方加速库API形参对应关系 |
| AccAPILink | 三方加速库API文档链接 |
| AscendAPILink | 昇腾API文档链接 |
| AscendLib | 推荐的昇腾API所在库 |

2. api-only模式下(源码扫描模式:--mode api-only)，会记录入口函数，入口函数体中API调用序列，匹配的三方加速库的序列功能，推荐的昇腾加速库的API调用序列和序列功能，以及推荐指数等，结果如下：

| Entry API     | Usr Call Seqs | Seq Labels | Recommended Sequences| Functional Description| Recommendation Index |    
| -------------- | ------------- | ---------- | -------------------- | --------------------- | --------------------- | 
| VEHICLE_RECG::VEHICLE_RECG(const std::__cxx11::string &) | cv::dnn::readNetFromCaffe(const cv::String &,const cv::String &)-->cv::dnn::readNetFromCaffe(const cv::String &,const cv::String &) | 1.加载caffe模型 | MxBase::Model::Model(std::string &,const int32_t) | 加载模型文件 | 1.0 |                                                                  
| VEHICLE_RECG::recognize(cv::Mat &,cv::Rect &) | cv::dnn::blobFromImage(cv::InputArray,double,const cv::Size &,const cv::Scalar &,bool,bool,int)-->cv::rectangle(cv::InputOutputArray,cv::Rect,const cv::Scalar &,int,int,int)-->cv::dnn::Net::setInput(cv::InputArray,const cv::String &,double,const cv::Scalar &)-->cv::dnn::Net::forward(const cv::String &)-->cv::Mat::reshape(int,int)-->cv::minMaxLoc(cv::InputArray,double *,double *,cv::Point *,cv::Point *,cv::InputArray) | 1.设置输入，进行模型推理 |  MxBase::Tensor::TensorMalloc(MxBase::Tensor &)-->MxBase::Model::Infer(std::vector<MxBase::Tensor> &,std::vector<MxBase::Tensor> &,MxBase::AscendStream &)| 给tensor分配空间，进行模型推理 | 1.0 |


输出数据说明：

| 标题                         | 说明                                                         |
| ----------------------------| ------------------------------------------------------------|
| Entry API                   | 入口API，以函数为入口API，比如main函数等                          |
| Usr Call Seqs               | 用户API调用序列，经过API关联之后的调用序列                         |
| Seq Labels                  | 在Usr Call Seqs中标红的子序列在加速库中对应的功能                  |
| Recommended Sequences       | 给Usr Call Seqs中标红的子序列推荐的昇腾加速库对应的序列             |
| Functional Description      | 昇腾加速库对应序列的功能                                        |
| Recommendation Index        | 推荐指数，[min_cfg,1]，最大值为1.0，值越大匹配度越高，为空表示没有推荐的序列 |