#!/bin/bash
# --------------------------------------------------
# 环境变量管理脚本 (直接 source 执行)
# 使用方式:
#   source msprechecker_env.sh    # 应用预期配置
#   source msprechecker_env.sh 0  # 还原为原始状态
# --------------------------------------------------

if [ "$1" = "0" ]; then
    export MINDIE_LOG_TO_FILE="<missing>"
    export MINDIE_LOG_TO_STDOUT="<missing>"
else
    export MINDIE_LOG_TO_FILE="1" # 建议将MindIE日志写入文件，便于排查问题
    export MINDIE_LOG_TO_STDOUT="1" # 建议将MindIE日志打屏，便于通过查看k8s日志查看程序运行状态
fi
