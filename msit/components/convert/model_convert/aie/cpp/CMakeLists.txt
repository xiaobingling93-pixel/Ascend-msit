# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
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

cmake_minimum_required(VERSION 3.0.0)
project(aie_convert VERSION 0.1.0)

#compile options
add_compile_options(-std=c++11)

IF(NOT DEFINED $ENV{CANN_HOME})
    set(ASCEND_HOME /usr/local/Ascend)
    set(ASCEND_TOOLKIT_HOME ${ASCEND_HOME}/ascend-toolkit/latest)
ENDIF()


add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

LINK_DIRECTORIES(
    $ENV{ASCENDIE_HOME}/lib
    $ENV{CANN_HOME}/x86_64-linux/lib64
    $ENV{CANN_HOME}/aarch64-linux/lib64
)

ADD_EXECUTABLE(aie_convert aie_convert.cpp)

TARGET_INCLUDE_DIRECTORIES(aie_convert PUBLIC
    $ENV{ASCENDIE_HOME}/include
)

TARGET_LINK_LIBRARIES(aie_convert PUBLIC ascendie)
