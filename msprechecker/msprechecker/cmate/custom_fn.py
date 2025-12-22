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

# ***************************************************************************************** #
# To define a custom function, just write its definition here, and use it in the cmate file #
# ***************************************************************************************** #

import os
import socket


def path_exists(path: str):
    try:
        return os.path.exists(path)
    except Exception:
        return False


def is_port_in_use(port: int, host: str = 'localhost', protocol: str = 'tcp'):
    protocol = protocol.lower()

    protocol_map = {
        'tcp': socket.SOCK_STREAM,
        'udp': socket.SOCK_DGRAM
    }

    if protocol not in protocol_map:
        raise ValueError
    
    sock_type = protocol_map[protocol]
    with socket.socket(socket.AF_INET, sock_type) as sock:
        if protocol.lower() == 'tcp':
            result = sock.connect_ex((host, port))
            return result == 0
        else:
            try:
                sock.bind((host, port))
                return False
            except Exception:
                return True
