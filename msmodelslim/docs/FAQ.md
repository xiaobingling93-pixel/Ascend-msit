# FAQ
## 1. 为什么我的程序会显示'Killed'并异常退出？

在使用msmodelslim工具运行推理量化时，出现类似以下报错信息：
```
Killed
...
[Error] TBE Subprocess[task_distribute] raise error[], main process disappeared!
...
```
请先确认你的进程没有被其他用户kill或抢占同一个NPU资源。一般而言，如果不存在其他用户抢占系统资源的情况，那么可能就是NPU显存不足或系统内存不足导致。可通过以下命令查看系统日志、看管系统内存情况、清理系统内存。
```shell
# dmesg查看被内核终止的进程或显存不足终止的进程
dmesg | grep -A 3 -B 1 -i "killed process\|oom-kill"

# 看管系统内存
watch free -h

# 清理缓存和内存，部分场景可能需要sudo权限
sync && echo 3 > /proc/sys/vm/drop_caches

# 停止所有python进程，部分场景可能需要sudo权限
pkill python
```
通过上述命令可以清理系统内存资源，此时再运行程序如果成功则问题解决，如果仍然异常退出，则需要升级系统资源。