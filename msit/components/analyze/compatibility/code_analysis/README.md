# 代码逻辑知识库：视图解析业务迁移优化

# 调优方向：应用开发高性能接口

# DVPP VPC接口支持多框多图
1. 在针对目标密集型场景，在模型处理前可通过批处理提升业务性能，具体分为两种情况，说明如下:
在对图像进行抠图、缩放、贴图、填充等处理时，AscendCL媒体数据处理部分提供了以下实现功能的接口:
一个接口只做一次操作（即单功能接口），例如acldvppVpcCropAsync、acldvppVpcResizeAsync、acldvppVpcMakeBorderAsync接口。
该方式下，如果想实现多个功能，例如抠图 + 缩放 + 填充，您需要调用以上3个接口。
一个接口做多个操作（即多功能组合接口），例如：acldvppVpcBatchCropResizePasteAsync、acldvppVpcBatchCropResizeMakeBorderAsync接口。
该方式下，如果想实现多个功能，例如抠图 + 缩放 + 填充，您仅需要调用1个接口acldvppVpcBatchCropResizeMakeBorderAsync。
同时，在对图像进行抠图、缩放、填充等处理时，AscendCL媒体数据处理部分提供了以下实现功能的接口:
一次处理一张图片，例如acldvppVpcCropAsync接口
该方式下，如果存在多张输入图片，一般都采用for循环的方式，针对每张图片，都调用一次acldvppVpcCropAsync接口。
一次处理多张图片（即批处理接口），例如acldvppVpcBatchCropAsync接口
该方式下，如果存在多张输入图片，只需调用一次acldvppVpcBatchCropAsync接口，而无需使用for循环重复调用单接口。
2. 针对上述可能出现的可优化之处，知识库需要对业务迁移中产生的数据进行分析。
3. 对此，用户需要获取迁移过程中产生的profiling数据，可以通过登录运行环境并进入“msprof”文件所在目录执行以下命令。
./msprof--application = /home/HwHiAiUser/HIAI_PROJECTS/MyAppname/out/main --output = /home/HwHiAiUser
此处application存放的是用户可执行文件所在的路径，output则是用户存放profiling数据所在的路径
详情可见profiling使用指南
4. 用户只需将获取的progiling数据置于知识库的../../data/profiling路径下即可通过知识库对该方向上可能出现的问题进行分析获取相应的反馈意见

# DVPP VDEC解码抽帧
1. 在昇腾710 AI处理器上，视频解码接口aclvdecSendFrame支持输出YUV420SP格式或RGB888格式，用户在使用过程中可设置接口参数输出不同的格式，省去调用acldvppVpcConvertColorAsync进行格式转换的步骤，减少接口调用。
若视频码流分辨率与模型输入图片的分辨率不一致，需要对解码后的图片进行缩放处理，也可以在视频解码接口aclvdecSendFrame中设置输出图片的分辨率，在解码的同时完成图片的缩放，省去单独调用缩放接口的步骤，减少接口调用。
总结下来，可以在视频解码接口aclvdecSendFrame中完成解码 + 缩放 + 色域转换三个功能，减少调用接口的数量，提升性能。
2. 视频解码是需要连续的数据，解码后的数据需要输出YUV格式，则每帧数据解码后，VDEC内部还需要通过VPC进行解压缩、缩放、格式转换的流程。
如果不想获取某一帧的解码结果， 可以调用aclvdecSendSkippedFrame接口，不需要申请输出内存，且VDEC内部也不通过VPC模块进行解压缩、缩放、格式转换的流程。减少内存申请，也减轻了VPC的处理压力，达到提升性能的目标。
3. 针对上述可能出现的可优化之处，知识库需要对业务迁移中产生的数据进行分析。
4. 对此，用户需要获取迁移过程中产生的profiling数据，方法同DVPP VPC接口支持多框多图中提到的步骤3
5. 用户只需将获取的progiling数据置于知识库的../../data/profiling路径下即可通过知识库对该方向上可能出现的问题进行分析获取相应的反馈意见

