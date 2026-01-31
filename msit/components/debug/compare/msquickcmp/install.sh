#!/bin/bash
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

CURRENT_DIR=$(dirname $(readlink -f $0))

build_om_so() {
  echo "Installing libsaveom.so"
  echo "This part is used for the accuracy comparison of mindir and onnx models. "
  echo "If can not install libsaveom.so, the usage of other components will not be affected."
  SITE_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")

  if [ "$ASCEND_TOOLKIT_HOME" != "" ]; then
      toolkit_home=$ASCEND_TOOLKIT_HOME
  else
      toolkit_home=$ASCEND_AICPU_PATH
  fi

  ge_dev_path=$toolkit_home/$(uname -m)-linux/
  if [ ! -e "$ge_dev_path/include" ] || [ ! -e "$ge_dev_path/lib64" ]; then
      echo "[WARNING] include or lib64 not found in ge_dev_path=$ge_dev_path, try installing CANN toolkit if comparing mindir and onnx models"
      return
  fi

  COMPILE_OPTIONS="-Wl,-z,relro,-z,now,-z,noexecstack -s -fstack-protector-all -ftrapv -D_FORTIFY_SOURCE=2"
  g++ ${CURRENT_DIR}/save_om_model/export_om_model.cpp \
          -I ${ge_dev_path}/include \
          -L ${ge_dev_path}/lib64 \
          -lge_compiler $COMPILE_OPTIONS \
          --std=c++11 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -o ${CURRENT_DIR}/libsaveom.so
  
  if [ ! -f "${CURRENT_DIR}/libsaveom.so" ]
  then
    echo "libsaveom.so compilation failed"
  else
    echo "libsaveom.so compilation successed"
    if [ ! -d "${SITE_PACKAGES_PATH}/msquickcmp/" ]
    then
      rm libsaveom.so
      echo "msquickcmp not exist, failed to install libsaveom.so"
    else
      if [ -f "${SITE_PACKAGES_PATH}/msquickcmp/libsaveom.so" ]
      then
        true
      else
        cp "${CURRENT_DIR}/libsaveom.so" "${SITE_PACKAGES_PATH}/msquickcmp/"
      fi
      echo "Finish libsaveom.so installation."
    fi
  fi
}


build_om_so
