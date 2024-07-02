# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from app_analyze.common.kit_config import KitConfig

global_filter = ['operator=', 'operator>>', 'operator<<', 'operator[]', 'operator()',
                 'operator!=', 'operator*', 'operator->', 'operator*=', 'operator++',
                 'operator&=', 'operator+=', 'operator>=', 'operator,', 'operator+',
                 'operator<', 'operator>', 'operator/', 'operator==', 'operator-',
                 'operator&', 'operator!']

_opencv_api_filter = ['cv::Mat.ptr', 'cv::Mat.at',
                      'getTickCount', 'getTickFrequency',
                      'line', 'circle', 'rectangle', 'putText', 'format',
                      'imshow', 'waitKey']
_opencv_file_filter = ['opencv2/core/utility.hpp',  # CPU性能统计
                       'opencv2/highgui.hpp',  # 高级GUI和媒体
                       'opencv2/calib3d.hpp',  # 相机校准和3D重建
                       'opencv2/photo.hpp',  # 用于处理和恢复照片
                       'opencv2/features2d.hpp',  # 特征点的探测和描述以及匹配
                       ]

_opencv_namespace_filter = {'cv::dnn::dnn4_v': 'cv::dnn'}

GLOBAL_FILTER_PREFIX = 'operator'
ACC_FILTER = {KitConfig.OPENCV: {'file_filter': _opencv_file_filter, 'api_filter': _opencv_api_filter,
                                 'namespace_filter': _opencv_namespace_filter}}
