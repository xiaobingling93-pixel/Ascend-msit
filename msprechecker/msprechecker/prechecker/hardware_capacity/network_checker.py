# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
import re
import json

from msprechecker.prechecker.register import PrecheckerBase, show_check_result, CheckResult
from msprechecker.prechecker.utils import (
    logger,
    SimpleProgressBar,
    parse_ranktable_file,
    get_interface_by_ip,
    run_shell_command,
)


class NetworkChecker(PrecheckerBase):
    PING_COMMAND = "ping -c 3 -q -W 2"
    KUBECTL_COMMAND = "kubectl get pods -A -o json"
    PING_RESULT_PATTERN = r"rtt min/avg/max/mdev = (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+) ms"
    PING_FAIL_MSG = " - Ping command failed, network may have issues."
    KUBECTL_FAIL_MSG = " - kubectl command failed"
    ERROR_VALUE = "error"

    @classmethod
    def ping_servers(cls, ips):
        ping_results = {}
        for ip in SimpleProgressBar(ips):
            ping_cmd = f"{cls.PING_COMMAND} {ip}"
            result = run_shell_command(ping_cmd, fail_msg=cls.PING_FAIL_MSG)

            if not result or result.returncode != 0:
                ping_results[ip] = cls.ERROR_VALUE
                continue

            if not result.stdout:
                ping_results[ip] = cls.ERROR_VALUE
                continue

            match = re.search(cls.PING_RESULT_PATTERN, result.stdout)
            if match:
                ping_results[ip] = float(match.group(2))
            else:
                ping_results[ip] = cls.ERROR_VALUE

        return ping_results

    @classmethod
    def filter_hostips_k8s(cls, data):
        hostips = []

        items = data.get("items", [])
        if not items:
            logger.debug("No 'items' field found in data or 'items' is empty")

        for item in data.get("items", []):
            phase = item.get("status", {}).get("phase")
            if phase != "Running":
                continue

            namespace = item.get("metadata", {}).get("namespace", "")
            if "mindie" not in namespace:
                continue

            labels = item.get("metadata", {}).get("labels", {})
            app = labels.get("app", "")
            if "mindie" not in app or "server" not in app:
                continue

            hostip = item.get("status", {}).get("hostIP")
            if hostip:
                hostips.append(hostip)

        return list(set(hostips))

    def collect_env(self, **kwargs):
        ranktable = parse_ranktable_file()
        
        if ranktable is None:
            # 处理 k8s 环境
            ips_k8s = run_shell_command(self.KUBECTL_COMMAND, self.KUBECTL_FAIL_MSG, print_error=False)
            
            # 检查命令执行是否成功
            if not ips_k8s: # k8s run failed
                logger.warning("kubectl command not available or failed, skipping k8s environment check")
                return None

            if not ips_k8s.stdout: # k8s does not return any message
                logger.warning("kubectl command does not return any pod's information")
                return None

            try:
                ips_k8s_dict = json.loads(ips_k8s.stdout)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from kubectl output: {e}")
                return None
            try:
                ranktable_map = self.filter_hostips_k8s(ips_k8s_dict)
            except Exception as e:
                logger.error(f"Unexpected error while getting k8s pods: {e}")
                return None
        else:
            server_list = ranktable.get("server_list", [])
            ranktable_map = [serv.get("server_id", None) for serv in server_list]
        try:
            other_ips = [ip for ip in ranktable_map if ip and not get_interface_by_ip(ip)[0]]
            return self.ping_servers(other_ips)
        except Exception as e:
            logger.error(f"Error during IP filtering or ping test: {e}")
            return None

    def do_precheck(self, envs, **kwargs):
        if not envs:
            logger.warning("No IPs available for network checking")
            return
        
        valid_results = {ip: result for ip, result in envs.items() if result != self.ERROR_VALUE}

        if not valid_results:
            logger.error("No valid ping results available for analysis")
            return
        
        ping_avg = sum(valid_results.values()) / len(valid_results)

        for server_ip, ping_time in envs.items():

            if ping_time == self.ERROR_VALUE:
                show_check_result(
                    "hardware",
                    "network_checker",
                    CheckResult.ERROR,
                    action=f"检查本机到服务器 {server_ip} 的连接状态",
                    reason=f"本机到对端卡的 ping 结果存在失败",
                )

            elif ping_time > ping_avg * 1.5:
                show_check_result(
                    "hardware",
                    "network_checker",
                    CheckResult.ERROR,
                    action=f"检查本机到服务器 {server_ip} 的连接状态",
                    reason=f"本机到对端卡的 ping 时间为 {ping_time} ms 超过平均时间50%",
                )
            
            else:
                show_check_result(
                    "hardware",
                    f"network_checker 本机到服务器{server_ip}的时间为{ping_time} ms",
                    CheckResult.OK,
                )

network_checker_instance = NetworkChecker()
