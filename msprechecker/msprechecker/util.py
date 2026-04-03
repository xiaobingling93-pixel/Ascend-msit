# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import dataclasses
import itertools
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
from enum import Enum
from typing import Optional

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_FORMAT = "%(levelname)-5s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"


logger = logging.getLogger(__name__)


class NpuType(Enum):
    d100 = "300"
    d500 = "300I_DUO"
    d801 = "800I A1"
    d802 = "800I A2"
    d803 = "800I A3"
    UNKNOWN = "unknown"


class ConnMode(Enum):
    ROUTE = "route"
    FIBER = "fiber"
    UNKNOWN = "unknown"


class Framework(Enum):
    MINDIE = "mindie"
    VLLM = "vllm"
    SGLANG = "sglang"
    UNKNOWN = "unknown"


class DeployMode(Enum):
    PD_MIX = "pd_mix"
    PD_DISAGGREGATION = "pd_disaggregation"
    PD_DISAGGREGATION_SINGLE_CONTAINER = "pd_disaggregation_single_container"
    EP = "ep"
    LWD = "lwd"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    npu_count: int
    npu_type: NpuType
    npu_memory_mb: Optional[int]
    conn_mode: ConnMode


def get_pkg_version(pkg_name: str) -> Optional[Version]:
    try:
        pkg_version = version(pkg_name)
    except PackageNotFoundError:
        return None

    try:
        return Version(pkg_version)
    except InvalidVersion:
        logger.warning(
            "Got invalid version '%s' from package '%s'", pkg_version, pkg_name
        )
        return None


def get_npu_count() -> int:
    """Count Davinci NPU character devices present on the machine."""
    template = "/dev/davinci{}"
    for device_id in itertools.count(0):
        try:
            mode = os.stat(template.format(device_id)).st_mode
        except OSError:
            # Device path doesn't exist: we've counted all devices.
            return device_id
        if not stat.S_ISCHR(mode):
            # Path exists but is not a character device: stop here.
            return device_id
    return 0  # unreachable; satisfies type checkers


def get_npu_type() -> NpuType:
    """Detect NPU type via lspci accelerator entries."""
    lspci = shutil.which("lspci")
    if not lspci:
        return NpuType.UNKNOWN

    try:
        output = subprocess.check_output([lspci], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        logger.exception("Failed to execute lspci")
        return NpuType.UNKNOWN

    device_pattern = re.compile(r"device\s*(\w+)", re.IGNORECASE)
    devices = []
    for line in output.splitlines():
        if "accelerator" not in line.lower():
            continue
        m = device_pattern.search(line)
        if m:
            devices.append(m.group(1))

    if not devices:
        return NpuType.UNKNOWN

    first = devices[0]
    if first not in NpuType.__members__ or any(d != first for d in devices):
        logger.debug("Inconsistent or unrecognised device types: %s", devices)
        return NpuType.UNKNOWN

    return NpuType[first]


def get_npu_memory() -> Optional[int]:
    """Return High-Bandwidth Memory capacity (MB) for device 0, or None if unavailable."""
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        return None

    try:
        output = subprocess.check_output(
            [npu_smi, "info", "-i", "0", "-t", "memory"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        logger.exception("Failed to execute npu-smi")
        return None

    for line in output.splitlines():
        if "HB" + "M Capacity" not in line or ":" not in line:
            continue
        value = line.split(":")[-1].strip()
        if value.isdigit():
            return int(value)

    return None


def get_conn_mode() -> ConnMode:
    """Detect NPU connection mode via hccn_tool LLDP output."""
    hccn_tool = shutil.which("hccn_tool")
    if not hccn_tool:
        return ConnMode.UNKNOWN

    try:
        output = subprocess.check_output(
            shlex.split(f"{hccn_tool} -i 0 -lldp -g"),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        logger.exception("Failed to execute hccn_tool")
        return ConnMode.UNKNOWN

    if not output:
        return ConnMode.UNKNOWN

    lines = output.splitlines()
    try:
        idx = lines.index("System Description TLV")
    except ValueError:
        return ConnMode.UNKNOWN

    if idx + 1 >= len(lines):
        return ConnMode.UNKNOWN

    next_line = lines[idx + 1]
    if "Routing" in next_line:
        return ConnMode.ROUTE
    if "AscendNPU" in next_line:
        return ConnMode.FIBER
    return ConnMode.UNKNOWN


def probe_hardware() -> HardwareProfile:
    """
    Probe all hardware characteristics in one call.

    The caller is responsible for caching the result if repeated
    subprocess calls are undesirable.
    """
    return HardwareProfile(
        npu_count=get_npu_count(),
        npu_type=get_npu_type(),
        npu_memory_mb=get_npu_memory(),
        conn_mode=get_conn_mode(),
    )
