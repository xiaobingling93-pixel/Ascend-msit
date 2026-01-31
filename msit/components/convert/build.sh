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

if [ ! "$(command -v pip3)" ];then
  echo "Error: pip3 is not installed."
  exit 1
fi

PIP3=$(readlink -f $(which pip3))
PIP3_DIR=$(dirname ${PIP3})
PYTHON=${PIP3_DIR}/python
if [ ! -f ${PYTHON} ];then
  PIP_HEAD=$(head -n 1 ${PIP3})
  PYTHON=${PIP_HEAD:2}
fi

MODEL_CONVERT_PATH=$(dirname $(${PYTHON} -c "import model_convert;print(model_convert.__file__)"))
CUR_PATH=$(dirname $(readlink -f $0))
