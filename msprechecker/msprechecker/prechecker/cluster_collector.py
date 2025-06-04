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
import os
import time
import socket
from collections import namedtuple

import torch
from msprechecker.prechecker.utils import parse_mindie_server_config, read_csv_or_json, logger
from msprechecker.prechecker.utils import get_local_to_master_ip, get_interface_by_ip
from msprechecker.prechecker.register import show_check_result, CheckResult

_DISTIBUT_ENVS = ["ranktable_map", "master_ip", "master_port", "local_ip", "rank", "interface", "world_size"]
DISTIBUT_ENVS = namedtuple("DISTIBUT_ENVS", _DISTIBUT_ENVS)(*_DISTIBUT_ENVS)
GLOBAL_DISTRIBUTE_COLLECTOR = None
GLOBAL_DISTRIBUTE_ENV = {}
DEFAULT_MASTER_PORT = 29400
MAX_SENDING_LEN = 40960


def get_rank_id_in_ranktable_by_ip(local_ip, rank_table):
    for rank_id, server_config in enumerate(rank_table.get("server_list", [])):
        if local_ip == server_config.get("server_id", None):
            return rank_id
    return None


def init_global_distribute_env(ranktable_file=None, service_config_path=None, master_ip=None):
    global GLOBAL_DISTRIBUTE_ENV
    global GLOBAL_LOCAL_IP
    global GLOBAL_LOCAL_INTERFACE
    if GLOBAL_DISTRIBUTE_ENV:
        return
    is_ranktable_file_available = ranktable_file and os.path.exists(ranktable_file)

    # Init master_ip firstly
    if not master_ip and not is_ranktable_file_available:
        logger.error("Neither master_ip nor ranktable_file provided. Will skip init_global_distribute_env")
        return

    ranktable = {} if not is_ranktable_file_available else read_csv_or_json(ranktable_file)
    master_ip = master_ip or ranktable.get("server_list", [{}])[0].get("server_id", None)
    if not master_ip:
        logger.error(f"Provided master_ip={master_ip} or parsed value from ranktable_file={ranktable_file} not valid")
        return

    # Init local_ip and interface after getting master_ip
    local_ip = get_local_to_master_ip(master_ip)
    interface, _ = get_interface_by_ip(local_ip)
    GLOBAL_DISTRIBUTE_ENV[DISTIBUT_ENVS.master_ip] = master_ip
    GLOBAL_DISTRIBUTE_ENV[DISTIBUT_ENVS.local_ip] = local_ip
    GLOBAL_DISTRIBUTE_ENV[DISTIBUT_ENVS.interface] = interface
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = interface
    logger.info(f"local_ip: {local_ip}, interface: {interface}")

    # Init rank and world_size by ranktable_file
    if not is_ranktable_file_available:
        logger.warning(
            f"ranktable_file={ranktable_file} is empty or not exists. "
            "Provide by env RANKTABLEFILE or argument --ranktable_file if needed."
        )
        return
    ranktable_map = {serv.get("server_id", None): rank for rank, serv in enumerate(ranktable.get("server_list", []))}
    if not ranktable_map:
        logger.error(f"ranktable_file={ranktable_file} is empty or not correctly set.")
        return
    if len(ranktable_map) < 2:
        logger.info(f"Only one server found in ranktable_file={ranktable_file}, skip distributed check")
        return
    if local_ip not in ranktable_map:
        logger.error(f"local_ip={local_ip } not exists in ranktable_file: {ranktable_file}.")
        return

    # Init master port by service_config_path
    master_port = None
    if service_config_path and os.path.exists(service_config_path):
        mindie_service_config = parse_mindie_server_config(service_config_path)
        master_port = mindie_service_config.get("ServerConfig", {}).get("port", None)
    if master_port is None:
        logger.warning(
            f"service_config_path not provided or port not set, will use default master port {DEFAULT_MASTER_PORT}"
        )
        master_port = DEFAULT_MASTER_PORT

    GLOBAL_DISTRIBUTE_ENV[DISTIBUT_ENVS.master_port] = master_port
    GLOBAL_DISTRIBUTE_ENV[DISTIBUT_ENVS.world_size] = len(ranktable_map)
    GLOBAL_DISTRIBUTE_ENV[DISTIBUT_ENVS.rank] = ranktable_map.get(local_ip, -1)
    GLOBAL_DISTRIBUTE_ENV[DISTIBUT_ENVS.ranktable_map] = ranktable_map

    logger.info(f"GLOBAL_DISTRIBUTE_ENV: {GLOBAL_DISTRIBUTE_ENV}")


class DistributeCollector:
    def __init__(self, master_ip=None, master_port=None, rank=None, world_size=None, backend="gloo"):
        self.master_ip = self._may_use_global_value(DISTIBUT_ENVS.master_ip, master_ip)
        self.master_port = self._may_use_global_value(DISTIBUT_ENVS.master_port, master_port)
        self.rank = self._may_use_global_value(DISTIBUT_ENVS.rank, rank)
        self.world_size = self._may_use_global_value(DISTIBUT_ENVS.world_size, world_size)

        self.backend = backend
        self.local_ip = GLOBAL_DISTRIBUTE_ENV.get(DISTIBUT_ENVS.local_ip, "127.0.0.1")
        self.init_method, self.is_dist_group_inited = f"tcp://{self.master_ip}:{self.master_port}", False
        logger.info(
            f"DistributeCollector: master_ip={self.master_ip}, master_port={self.master_port}, rank={self.rank}, "
            f"world_size={self.world_size}"
        )

    @staticmethod
    def _may_use_global_value(key, value=None):
        return GLOBAL_DISTRIBUTE_ENV[key] if not value and key in GLOBAL_DISTRIBUTE_ENV else value
    
    @staticmethod
    def _get_all_ips():
        if "ranktable_map" in GLOBAL_DISTRIBUTE_ENV:
            return list(GLOBAL_DISTRIBUTE_ENV["ranktable_map"].keys())
        
        ranktable_file = os.getenv('RANK_TABLE_FILE') or os.getenv('RANKTABLEFILE')
        if ranktable_file and os.path.exists(ranktable_file):
            ranktable = read_csv_or_json(ranktable_file)
            ranktable_map = {serv.get("server_id", None): rank 
                             for rank, serv in enumerate(ranktable.get("server_list", []))}
            return list(ranktable_map.keys())
        
        return []

    def gather(self, contents):
        if not all([isinstance(self.world_size, int), 
                   isinstance(self.rank, int),
                   self.master_ip,
                   self.master_port]):
            logger.error("Missing required parameters for distributed gathering")
            return None
        
        if not self._check_connectivity():
            logger.error("Network connectivity check failed. Skip gathering.")
            return None

        if not self.is_dist_group_inited:
            if not isinstance(self.world_size, int) or not isinstance(self.rank, int):
                logger.error(f"world_size and rank not set. Got world_size={self.world_size}, rank={self.rank}")
                return None
            torch.distributed.init_process_group(
                backend=self.backend, init_method=self.init_method, world_size=self.world_size, rank=self.rank
            )
            self.is_dist_group_inited = True

        combined_str = f"{contents}@{self.local_ip}"

        if isinstance(contents, str):
            bytes_contents = contents.encode()
        elif isinstance(contents, bytes):
            bytes_contents = contents
        else:
            logger.error(f"contents of type {type(contents).__name__} not supported in DistributeCollector")
            return None

        combined_str = f"{contents}@{self.local_ip}"
        byte_data = combined_str.encode()[:MAX_SENDING_LEN].ljust(MAX_SENDING_LEN, b"\x00")  # padding
        tensor_data = torch.tensor(list(byte_data), dtype=torch.uint8)

        if self.rank == 0:
            gather_list = [torch.zeros_like(tensor_data) for _ in range(self.world_size)]
            torch.distributed.gather(tensor_data, gather_list=gather_list, dst=0)
        else:
            torch.distributed.gather(tensor_data, gather_list=None, dst=0)

        result = {}
        if self.rank == 0:
            for idx, tensor in enumerate(gather_list):
                byte_result = bytes(tensor.numpy().tobytes())
                decoded_str = byte_result.decode().split("\x00", 1)[0]
                if "@" in decoded_str:
                    content, ip = decoded_str.rsplit("@", 1)
                    result[ip] = content
                else:
                    result[f"unknown_{idx}"] = decoded_str

        torch.distributed.destroy_process_group()
        return result if self.rank == 0 else None

    def _check_connectivity(self):
        all_ips = self._get_all_ips()
        if not all_ips:
            logger.warning("No valid IPs found for connectivity check")
            return False
            
        if self.local_ip not in all_ips:
            logger.error(f"Current IP {self.local_ip} not in cluster IPs: {all_ips}")
            return False
            
        if self.rank == 0:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
            try:
                server_socket.bind((self.local_ip, self.master_port))
                server_socket.listen(self.world_size - 1)
                logger.info(f"Master started temporary server on {self.local_ip}:{self.master_port}")
                
                connected_ips = set()
                start_time = time.time()
                timeout = 30
                
                while len(connected_ips) < len(all_ips) - 1 and time.time() - start_time < timeout:
                    try:
                        server_socket.settimeout(5)
                        client_sock, (client_ip, _) = server_socket.accept()

                        if client_ip in all_ips:
                            logger.info(f"Connected to cluster node: {client_ip}")
                            connected_ips.add(client_ip)
                        else:
                            logger.warning(f"Rejected connection from unauthorized IP: {client_ip}")
                            
                        client_sock.close()
                    except socket.timeout:
                        continue
      
                missing_ips = set(all_ips) - {self.local_ip} - connected_ips
                if missing_ips:
                    show_check_result(
                        "Cluster", "Connectivity", CheckResult.ERROR,
                        action="请检查防火墙状态（sudo systemctl status firewalld）、交换机的 acl 配置以及端口安全策略是否合适",
                        reason=f"以下节点未连接: {', '.join(missing_ips)}"
                    )
                    return False
                    
                return True
                                          
            except OSError as e:
                show_check_result(
                    "Cluster", "Connectivity", CheckResult.ERROR,
                    action="请检查防火墙状态（sudo systemctl status firewalld）、交换机的 acl 配置以及端口安全策略是否合适。或者通过 `-port` 更换端口",
                    reason=f"防火墙可能阻止了端口，或者端口已被使用，错误信息: {str(e)}"
                )
                return False
            finally:
                server_socket.close()
                
        else:
            for attempt in range(3):
                try:
                    with socket.create_connection(
                        (self.master_ip, self.master_port), 
                        timeout=10
                    ) as sock:
                        logger.info(f"Successfully connected to master {self.master_ip}:{self.master_port}")
                        return True
                except (ConnectionRefusedError, socket.timeout, socket.gaierror) as e:
                    logger.warning(f"Connection attempt {attempt+1}/3 failed: {str(e)}")
                    time.sleep(2)
            
            show_check_result(
                "Cluster", "Connectivity", CheckResult.ERROR,
                action="请检查防火墙状态（sudo systemctl status firewalld）、交换机的 acl 配置以及端口安全策略是否合适",
                reason=f"无法连接到主节点 {self.master_ip}:{self.master_port}"
            )
            return False


def distribute_collector(contents, master_ip=None, master_port=None, rank=None, world_size=None):
    global GLOBAL_DISTRIBUTE_COLLECTOR
    if GLOBAL_DISTRIBUTE_COLLECTOR is None:
        GLOBAL_DISTRIBUTE_COLLECTOR = DistributeCollector(
            master_ip=master_ip, master_port=master_port, rank=rank, world_size=world_size
        )
    return GLOBAL_DISTRIBUTE_COLLECTOR.gather(contents)
