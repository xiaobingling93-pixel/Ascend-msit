# -*- coding: utf-8 -*-
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
import os
import stat
import json
import shlex
import itertools
import subprocess
from enum import Enum
from abc import ABC, abstractmethod

from msguard.security import open_s
from .log import global_logger


class NpuType(Enum):
    TP_300 = "d100"
    TP_300I_DUO = "d500"
    TP_A1 = "d801"
    TP_A2 = "d802"
    TP_A3 = "d803"

    @property
    def display(self):
        return {
            NpuType.TP_300: "300",
            NpuType.TP_300I_DUO: "300I Duo",
            NpuType.TP_A1: "800I A1",
            NpuType.TP_A2: "800I A2",
            NpuType.TP_A3: "800I A3",
        }[self]


def get_npu_count():
    davinci_path_template = "/dev/davinci{}"

    for device_id in itertools.count(0):
        device_path = davinci_path_template.format(device_id)
        try:
            f_mode = os.stat(device_path).st_mode
        except Exception:
            break

        if not stat.S_ISCHR(f_mode):
            break

    return device_id


def get_npu_type():
    try:
        output = subprocess.check_output(['/usr/bin/lspci'], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None, None

    device_pattern = re.compile(r'device\s*(\w+)', re.IGNORECASE)
    device_list = []
    zip_engine_nums = 0
    for line in output.split('\n'):
        if 'accelerator' in line:
            m = device_pattern.search(line)
            if not m:
                zip_engine_nums += 1 # d803 contains extra zip engines
                continue
            device_list.append(m.group(1))
    
    if not device_list:
        return None, None

    if any(device != device_list[0] for device in device_list):
        return None, None
    
    npu_type = device_list[0]
    if npu_type not in (member.value for member in NpuType):
        return None, None

    return NpuType(npu_type), len(device_list) - zip_engine_nums


def get_conn_mode():
    lldp_cmd = "hccn_tool -i 0 -lldp -g"
    try:
        output = subprocess.check_output(shlex.split(lldp_cmd), stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    
    if not output:
        return None
    
    fields = output.split('\n')
    tlv_desc = 'System Description TLV'
    if tlv_desc not in fields:
        return None
    
    desc_idx = fields.index(tlv_desc)
    if desc_idx >= len(fields):
        return None
    
    route_fields = "Routing"
    fiber_fields = "AscendNPU"
    if route_fields in fields[desc_idx + 1]:
        return "route"
    
    if fiber_fields in fields[desc_idx + 1]:
        return "fiber"
    
    return None


class RankTableParser(ABC):
    def __init__(self, rank_table):
        if isinstance(rank_table, str):
            self.rank_table = self.load(rank_table)
        elif isinstance(rank_table, dict):
            self.rank_table = rank_table
        else:
            raise TypeError(f"'rank_table' expected to be str or dict. Got {type(rank_table).__name__} instead.")

    @staticmethod
    def load(rank_table_path: str):
        with open_s(rank_table_path) as f:
            return json.load(f)

    @abstractmethod
    def parse(self):
        pass


class A2RankTableParser(RankTableParser):
    def parse(self):
        ip_to_rank_id = {}
        for server in self.rank_table.get("server_list", []):
            server_id = server["server_id"]
            rank_id_to_device_ip = {}
            ip_to_rank_id[server_id] = rank_id_to_device_ip

            for device in server.get("device", []):
                rank_id = device.get("rank_id")
                device_ip = device.get("device_ip")
                rank_id_to_device_ip[rank_id] = device_ip

        return ip_to_rank_id


class A3RankTableParser(RankTableParser):
    def parse(self):
        pass


def get_rank_table_parser() -> RankTableParser:
    npu_type_to_parser = {
        NpuType.TP_A2: A2RankTableParser,
        NpuType.TP_A3: A3RankTableParser
    }

    npu_type, _ = get_npu_type()
    if not npu_type:
        npu_type = NpuType.TP_A2
        global_logger.warning("Auto-detect npu device failed, set to '%s' as a fall back", npu_type.display)
    
    elif npu_type not in npu_type_to_parser:
        npu_type = NpuType.TP_A2
        global_logger.warning(
            "No appropriate rank table parser found for current npu type (%s), using 'A2' format instead.", 
            npu_type.display
        )

    return npu_type_to_parser.get(npu_type)
