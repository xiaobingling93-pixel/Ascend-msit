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

from __future__ import annotations

import shutil
import typing
import traceback as tb

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from colorama import Fore, Style
from packaging.version import InvalidVersion, Version

from ..util import (
    ConnMode,
    CONTAINER_CPU_HIGH_PERF_AMBIGUITY_HINT,
    DeployMode,
    detect_framework,
    Framework,
    get_conn_mode,
    get_npu_count,
    get_npu_memory,
    is_in_container,
    parse_rank_table,
    parse_version_heuristic,
)
from .strategy import (
    CommandStatus,
    CPU,
    CPUHighPerformance,
    Driver,
    JeMalloc,
    Kernel,
    Network,
    NetworkProbe,
    NPU,
    TransparentHugepage,
    VirtualMachine,
)


class Severity(IntEnum):
    """
    Failure severity.

    Attributes:
        INFO: Suboptimal but not blocking.
        WARNING: Possible issue, worth investigating.
        ERROR: Blocking, the deployment should not proceed.

    Examples:
        if outcome.severity >= Severity.WARNING:
            log.warning(...)
    """

    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True)
class Passed:
    """The check ran and found no problems."""

    result_text: str = "ok"


@dataclass(frozen=True)
class Skipped:
    """The check was intentionally not run (e.g. not applicable to this scenario)."""

    reason: str


@dataclass(frozen=True)
class Failed:
    """The check detected a problem.

    Attributes:
        msg: The error message.
        severity: The severity of the error.
        result_text: The text to display in the result.
        traceback: The traceback of the error.

    Examples:
        Failed(
            msg="The check detected a problem.",
            severity=Severity.ERROR,
            result_text="failed",
            traceback="Traceback of the error."
        )
    """

    msg: str
    severity: Severity
    result_text: str
    traceback: Optional[str] = None


#: Union of all possible check outcomes.  Use isinstance() to discriminate.
CheckOutcome = Union[Passed, Skipped, Failed]


@dataclass(frozen=True)
class CheckGroup:
    """
    A logical section that groups related checks in the report.

    Attributes:
        key:   Machine-readable identifier (e.g. ``"system"``).
        title: Human-readable section header shown in the collection log.
    """

    key: str
    title: str


@dataclass(frozen=True)
class Check:
    """
    A single executable check.

    The ``fn`` callable takes no arguments; any runtime parameters
    (framework, scene, thresholds, …) must be captured via closure or
    ``functools.partial`` at construction time in ``suite.py``.

    Attributes:
        description: One-line label shown next to the result in the report.
        fn:          Zero-argument callable returning a ``CheckOutcome``.
    """

    description: str
    fn: Callable[[], CheckOutcome]


@dataclass
class CheckRecord:
    """Pairs a ``Check`` with the ``CheckOutcome`` produced when it ran."""

    check: Check
    outcome: CheckOutcome

    @property
    def passed(self) -> bool:
        return isinstance(self.outcome, Passed)

    @property
    def skipped(self) -> bool:
        return isinstance(self.outcome, Skipped)

    @property
    def failed(self) -> bool:
        return isinstance(self.outcome, Failed)


def has_errors(records: List[CheckRecord]) -> bool:
    """Return True if any record failed with ERROR severity."""
    return any(r.failed and r.outcome.severity == Severity.ERROR for r in records)


@dataclass(frozen=True)
class CheckContext:
    """Immutable snapshot of the deployment environment, passed to every check."""

    framework: Framework
    deploy_mode: DeployMode
    rank_table_path: Optional[Path] = None
    hardware: bool = False
    threshold: int = 20
    _shared: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def has_rank_table(self) -> bool:
        return self.rank_table_path is not None

    def needs_hardware(self) -> bool:
        return self.hardware

    def is_vllm(self) -> bool:
        return self.framework == Framework.VLLM

    def network_results(self) -> Dict[str, Any]:
        """Collect network probe results (memoised across all network checks)."""
        if "network" not in self._shared:
            framework = detect_framework()
            rank_table = parse_rank_table(Path(self.rank_table_path), framework)
            npu_count = get_npu_count()
            self._shared["network"] = Network(
                rank_table=rank_table, npu_count=npu_count
            ).execute()
        return self._shared["network"]


CheckFn = Callable[[CheckContext], CheckOutcome]


@dataclass(frozen=True)
class _Entry:
    """"""

    name: str
    fn: CheckFn


SYSTEM = CheckGroup("system", "Checking system environment")
ASCEND = CheckGroup("ascend", "Checking Ascend environment")
NETWORK = CheckGroup("network", "Checking network connectivity")
STRESS = CheckGroup("stress", "Running hardware stress tests")
EP = CheckGroup(
    "expert parallelism", "Checking expert parallelism prerequisites and environment"
)


_registry: Dict[CheckGroup, List[_Entry]] = {}
for _g in (SYSTEM, ASCEND, NETWORK, STRESS, EP):
    _registry[_g] = []


def check(
    name: str,
    *,
    group: CheckGroup,
) -> Callable[[CheckFn], CheckFn]:
    """Register a check function with signature ``(ctx) -> CheckOutcome``."""

    def decorator(fn: CheckFn) -> CheckFn:
        try:
            hints = typing.get_type_hints(fn)
        except Exception as e:
            raise ValueError(
                f"Check function {fn.__qualname__} has unresolvable annotations: {e}"
            ) from e

        if "return" not in hints:
            raise ValueError(
                f"Check function {fn.__qualname__} has no return annotation"
            )
        if hints["return"] is not CheckOutcome:
            raise ValueError(
                f"Check function {fn.__qualname__} return type is not CheckOutcome, "
                f"got {hints['return']!r}"
            )
        if "ctx" not in hints:
            raise ValueError(
                f"Check function {fn.__qualname__} has no 'ctx' parameter annotation"
            )
        if hints["ctx"] is not CheckContext:
            raise ValueError(
                f"Check function {fn.__qualname__} ctx type is not CheckContext, "
                f"got {hints['ctx']!r}"
            )

        _registry[group].append(_Entry(name=name, fn=fn))
        return fn

    return decorator


# System Checks ─────────────────────────────────────────────────────────────────
@check("CPU high performance mode", group=SYSTEM)
def _cpu_high_performance(ctx: CheckContext) -> CheckOutcome:
    if CPUHighPerformance().execute():
        return Passed("on")
    if is_in_container():
        return Failed(
            msg=CONTAINER_CPU_HIGH_PERF_AMBIGUITY_HINT,
            severity=Severity.WARNING,
            result_text="unknown",
        )
    return Failed(
        msg="建议开启 CPU 高性能模式，可使 CPU 运行在最大频率，提升性能，但会增加能耗",
        severity=Severity.INFO,
        result_text="off",
    )


@check("Virtual machine detection", group=SYSTEM)
def _virtual_machine(ctx: CheckContext) -> CheckOutcome:
    if VirtualMachine().execute():
        return Failed(
            msg="检测到虚拟机环境，其 CPU 核数和频率与物理机有差异，可能导致性能下降",
            severity=Severity.INFO,
            result_text="virtual",
        )
    return Passed("physical")


@check("Transparent hugepage", group=SYSTEM)
def _transparent_hugepage(ctx: CheckContext) -> CheckOutcome:
    result = TransparentHugepage().execute()
    if not (result and "[always]" in result):
        return Failed(
            msg="建议开启透明大页（Transparent Hugepage），可使吞吐率结果更加稳定",
            severity=Severity.INFO,
            result_text="disabled",
        )
    return Passed("enabled")


@check("Kernel version", group=SYSTEM)
def _kernel_version(ctx: CheckContext) -> CheckOutcome:
    info = Kernel().execute() or {}
    release = info.get("release")

    if not release:
        return Failed(
            msg="无法检测到内核版本号",
            severity=Severity.WARNING,
            result_text="unknown",
        )

    try:
        version = parse_version_heuristic(release)
    except InvalidVersion:
        return Failed(
            msg=f"无法解析内核版本号: {release}",
            severity=Severity.WARNING,
            result_text="unknown",
            traceback=tb.format_exc(),
        )

    min_version = Version("5.10")
    if version < min_version:
        return Failed(
            msg=(
                f"当前内核版本 {release} 低于推荐版本 5.10，"
                "升级后 CPU 指令下发速度更快，可减少 host bound"
            ),
            severity=Severity.INFO,
            result_text=f"{release} (requires >= {min_version})",
        )
    return Passed(release)


@check("NPU connection mode", group=SYSTEM)
def _npu_connection_mode(ctx: CheckContext) -> CheckOutcome:
    conn_mode = get_conn_mode()
    if conn_mode == ConnMode.FIBER:
        return Failed(
            msg=(
                '检测到网线对端设备为昇腾 NPU，疑似"双机背靠背直连"架构。'
                "该架构下 HCCL 不支持全互联通信链路自动建立，模型通信可能受到影响，"
                "请确认当前部署环境"
            ),
            severity=Severity.WARNING,
            result_text="fiber direct",
        )
    return Passed(conn_mode.value)


@check("jemalloc", group=SYSTEM)
def _jemalloc(ctx: CheckContext) -> CheckOutcome:
    if ctx.framework != Framework.VLLM:
        return Skipped(reason="only check for VLLM")

    if not JeMalloc().execute():
        return Failed(
            msg=(
                "未检测到通过 apt/yum 安装的 jemalloc，建议安装以提升性能；"
                "若已安装至自定义路径，请忽略此项"
            ),
            severity=Severity.INFO,
            result_text="not installed",
        )
    return Passed("installed")


# Ascend Checks ─────────────────────────────────────────────────────────────────
@check("Driver version", group=ASCEND)
def _driver_version(ctx: CheckContext) -> CheckOutcome:
    min_default = Version("24.1")
    min_vllm_ep = Version("25.0")

    is_vllm_ep = ctx.framework == Framework.VLLM and ctx.deploy_mode == DeployMode.EP
    min_ver = min_vllm_ep if is_vllm_ep else min_default
    severity = Severity.ERROR if is_vllm_ep else Severity.INFO

    driver_info = Driver().execute()
    if driver_info is None:
        return Failed(
            msg="无法检测到 Ascend 驱动，请确认驱动已正确安装",
            severity=Severity.WARNING,
            result_text="not found",
        )

    version_str = driver_info.get("Version", "")
    try:
        version = parse_version_heuristic(version_str)
    except InvalidVersion:
        return Failed(
            msg=f"驱动版本号格式无法解析: {version_str!r}，请检查驱动安装状态",
            severity=Severity.WARNING,
            result_text="unknown",
        )

    if version < min_ver:
        min_str = f"{min_ver.major}.{min_ver.minor}"
        msg = (
            f"当前驱动版本 {version_str} 低于 VLLM 大 EP 场景所需最低版本 {min_str}，"
            "不升级可能导致 dispatch_combine 等算子失败"
            if is_vllm_ep
            else f"当前驱动版本 {version_str} 低于推荐版本 {min_str}，"
            "建议升级至最新驱动以获得更好的性能"
        )
        return Failed(
            msg=msg,
            severity=severity,
            result_text=f"{version_str} (requires >= {min_str})",
        )
    return Passed(version_str)


# Network Checks ─────────────────────────────────────────────────────────────────
@check("Network ping test", group=NETWORK)
def _ping(ctx: CheckContext) -> CheckOutcome:
    if ctx.rank_table_path is None:
        return Skipped("no rank table path is supplied")

    results = ctx.network_results()
    ping_results: Dict[str, Optional[NetworkProbe]] = {
        k: v
        for k, v in results.items()
        if k.startswith("ping_")
    }

    if not ping_results:
        return Skipped("no ping results in rank table output")

    if all(v is None for v in ping_results.values()):
        return Failed(
            msg="当前环境没有 ping 命令，无法检测网络连通性",
            severity=Severity.ERROR,
            result_text="ping command not found",
        )

    failures: List[Tuple[str, Optional[NetworkProbe]]] = []
    for key, res in ping_results.items():
        label = key.replace("ping_", "", 1)
        if res is None:
            failures.append((label, None))
        elif res.result != CommandStatus.SUCCESS:
            failures.append((label, res))

    if failures:
        groups = "\n".join(
            [
                (
                    f"{label}: ping 命令不存在"
                    if result is None
                    else f"{label}: {result.stderr or result.stdout or result.result.value}"
                )
                for label, result in failures
            ]
        )
        return Failed(
            msg=(
                "Rank Table 中的 Host IP 需要互相 ping 通，当前有以下主机 ping 失败：\n"
                f"{groups}"
            ),
            severity=Severity.ERROR,
            result_text=f"{len(failures)} failed",
        )
    return Passed("all hosts ping successfully")


@check("Network VNIC status", group=NETWORK)
def _vnic_status(ctx: CheckContext) -> CheckOutcome:
    if ctx.rank_table_path is None:
        return Skipped("no rank table path is supplied")

    results = ctx.network_results()
    if "vnic" not in results:
        return Skipped("vnic result not present")
    vnic_results = results["vnic"]
    if vnic_results is None:
        return Failed(
            msg="没有 hccn_tool 工具，无法检测 VNIC 状态",
            severity=Severity.ERROR,
            result_text="hccn_tool not found",
        )

    if not isinstance(vnic_results, list):
        raise ValueError("vnic results must be a list")

    failures: List[Tuple[int, Optional[NetworkProbe]]] = []
    for i, result in enumerate(vnic_results):
        if result is None:
            failures.append((i, None))
        elif result.result != CommandStatus.SUCCESS or "UP" not in result.stdout:
            failures.append((i, result))

    if failures:
        groups = "\n".join(
            [
                (
                    f"Device {i}: hccn_tool 命令不存在"
                    if result is None
                    else f"Device {i}: {result.stderr or result.stdout or result.result.value}"
                )
                for i, result in failures
            ]
        )
        return Failed(
            msg=(
                "当前机器的每个 NPU 设备的 VNIC 状态需要为 UP, 当前有以下设备 VNIC 状态存在异常：\n"
                f"{groups}"
            ),
            severity=Severity.ERROR,
            result_text=f"{len(failures)} failed",
        )
    return Passed("all NPU devices' VNIC status are UP")


@check("Network link status", group=NETWORK)
def _link_status(ctx: CheckContext) -> CheckOutcome:
    if ctx.rank_table_path is None:
        return Skipped("no rank table path is supplied")

    results = ctx.network_results()
    if "link" not in results:
        return Skipped("link result not present")
    link_results = results["link"]
    if link_results is None:
        return Failed(
            msg="没有 hccn_tool 工具，无法检测链路状态",
            severity=Severity.ERROR,
            result_text="hccn_tool not found",
        )

    if not isinstance(link_results, list):
        raise ValueError("link results must be a list")

    failures: List[Tuple[int, Optional[NetworkProbe]]] = []
    for i, result in enumerate(link_results):
        if result is None:
            failures.append((i, None))
        elif result.result != CommandStatus.SUCCESS or "UP" not in result.stdout:
            failures.append((i, result))

    if failures:
        groups = "\n".join(
            [
                (
                    f"Device {i}: hccn_tool 命令不存在"
                    if result is None
                    else f"Device {i}: {result.stderr or result.stdout or result.result.value}"
                )
                for i, result in failures
            ]
        )
        return Failed(
            msg=(
                "当前机器的每个 NPU 设备的链路状态需要为 UP，当前有以下设备链路状态存在异常：\n"
                f"{groups}"
            ),
            severity=Severity.ERROR,
            result_text=f"{len(failures)} failed",
        )
    return Passed("all NPU devices' link status are UP")


@check("TLS link status", group=NETWORK)
def _tls_status(ctx: CheckContext) -> CheckOutcome:
    if ctx.rank_table_path is None:
        return Skipped("no rank table path is supplied")

    results = ctx.network_results()
    if "tls" not in results:
        return Skipped("tls result not present")
    tls_results = results["tls"]
    if tls_results is None:
        return Failed(
            msg="没有 hccn_tool 工具，无法检测 TLS 状态",
            severity=Severity.ERROR,
            result_text="hccn_tool not found",
        )

    if not isinstance(tls_results, list):
        raise ValueError("tls results must be a list")

    failures: List[Tuple[int, Optional[NetworkProbe]]] = []
    for i, result in enumerate(tls_results):
        if result is None:
            failures.append((i, None))
        elif (
            result.result != CommandStatus.SUCCESS
            or "tls switch[0]" not in result.stdout
        ):
            failures.append((i, result))

    if failures:
        groups = "\n".join(
            [
                (
                    f"Device {i}: hccn_tool 命令不存在"
                    if result is None
                    else f"Device {i}: {result.stderr or result.stdout or result.result.value}"
                )
                for i, result in failures
            ]
        )
        return Failed(
            msg=(
                "当前机器的每个 NPU 设备的 TLS 需要关闭，当前有以下设备 TLS 状态存在异常：\n"
                f"{groups}"
            ),
            severity=Severity.ERROR,
            result_text=f"{len(failures)} failed",
        )
    return Passed("all NPU devices' TLS are closed")


@check("HCCL ping test", group=NETWORK)
def _hccl_ping(ctx: CheckContext) -> CheckOutcome:
    if ctx.rank_table_path is None:
        return Skipped("no rank table path is supplied")

    results = ctx.network_results()
    if "hccl_ping" not in results and "hccs_ping" not in results:
        return Skipped("hccl/hccs ping result not present")
    hccl_results = (
        results["hccl_ping"] if "hccl_ping" in results else results["hccs_ping"]
    )
    if hccl_results is None:
        return Failed(
            msg="没有 hccn_tool 工具，无法检测 HCCL/HCCS ping 状态",
            severity=Severity.ERROR,
            result_text="hccn_tool not found",
        )

    if not isinstance(hccl_results, list):
        raise ValueError("hccl/hccs ping results must be a list")

    failures: List[Tuple[int, str, Optional[NetworkProbe]]] = []
    for device_id, probe_results in enumerate(hccl_results):
        if probe_results is None:
            failures.append((device_id, "*", None))
            continue
        if not isinstance(probe_results, dict):
            raise ValueError("hccl/hccs ping probe results must be a dict")

        for peer_ip, result in probe_results.items():
            if result is None:
                failures.append((device_id, peer_ip, None))
            elif result.result != CommandStatus.SUCCESS:
                failures.append((device_id, peer_ip, result))

    if failures:
        groups = "\n".join(
            [
                (
                    f"Device {device_id} -> {peer_ip}: hccn_tool 命令不存在"
                    if result is None
                    else f"Device {device_id} -> {peer_ip}: {result.stderr or result.stdout or result.result.value}"
                )
                for device_id, peer_ip, result in failures
            ]
        )
        return Failed(
            msg=(f"HCCL/HCCS ping 检查未通过，当前有以下设备 ping 失败：\n{groups}"),
            severity=Severity.ERROR,
            result_text=f"{len(failures)} failed",
        )
    return Passed("all NPU devices' HCCL/HCCS ping are successful")


# Stress Checks ─────────────────────────────────────────────────────────────────
def _find_abnormal(
    results: dict[int, Optional[float]], threshold_pct: int
) -> list[int]:
    valid = {k: v for k, v in results.items() if v is not None and v > 0}
    if not valid:
        return []
    mean = sum(valid.values()) / len(valid)
    if mean == 0:
        return []
    ratio = threshold_pct / 100.0
    return [uid for uid, t in valid.items() if t > mean * (1 + ratio)]


@check("CPU stress test", group=STRESS)
def _cpu_stress(ctx: CheckContext) -> CheckOutcome:
    if not ctx.hardware:
        return Skipped("hardware stress tests are not enabled")

    raw = CPU().execute()
    if raw is None:
        return Failed(
            msg="CPU 压测无法完成 (torch not available)",
            severity=Severity.WARNING,
            result_text="skipped",
        )
    abnormal = _find_abnormal(raw, ctx.threshold)
    if abnormal:
        return Failed(
            msg=(
                f"检测到异常核: {', '.join(map(str, abnormal))}。"
                "这些核心的计算时间明显高于其他核心，请检查系统负载或硬件问题"
            ),
            severity=Severity.ERROR,
            result_text="failed",
        )
    return Passed("ok")


@check("NPU stress test", group=STRESS)
def _npu_stress(ctx: CheckContext) -> CheckOutcome:
    if not ctx.hardware:
        return Skipped("hardware stress tests are not enabled")

    raw = NPU().execute()
    if raw is None:
        return Failed(
            msg="NPU 压测无法完成，因为 torch 或者 torch_npu 不可用",
            severity=Severity.WARNING,
            result_text="skipped",
        )
    abnormal = _find_abnormal(raw, ctx.threshold)
    if abnormal:
        return Failed(
            msg=(
                f"检测到异常设备: {', '.join(map(str, abnormal))}。"
                "这些设备的计算时间明显高于其他设备，请检查系统负载或硬件问题"
            ),
            severity=Severity.ERROR,
            result_text="failed",
        )
    return Passed("ok")


# Suite Assembly ─────────────────────────────────────────────────────────────────
def build_suite(
    *,
    framework: Framework,
    deploy_mode: DeployMode,
    rank_table_path: Optional[Path] = None,
    hardware: bool = False,
    threshold: int = 20,
) -> Dict[CheckGroup, List[Check]]:
    """
    Assemble checks grouped by section.

    Args:
        framework: The framework to be checked (MindIE, VLLM, SGLANG, etc).
        deploy_mode: The deployment mode (e.g. pd_mix, pd_disaggregation_ep).
        rank_table_path: Path to the rank table file; enables network checks. Defaults to None.
        hardware: Whether to run hardware stress tests. Defaults to False.
        threshold: Abnormality threshold percentage for stress tests (default 20). Defaults to 20.

    Returns:
        A dictionary of check groups and their checks.
    """
    ctx = CheckContext(
        framework=framework,
        deploy_mode=deploy_mode,
        rank_table_path=rank_table_path,
        hardware=hardware,
        threshold=threshold
    )

    suite: Dict[CheckGroup, List[Check]] = {}

    for group, entries in _registry.items():
        suite[group] = [Check(e.name, lambda _e=e: _e.fn(ctx)) for e in entries]

    return suite


class PrecheckRunner:
    """
    Execute a grouped suite of checks and render a two-phase report.

    Accepts the ``Dict[CheckGroup, List[Check]]`` returned by
    ``build_suite``.
    """

    _SEV_COLOR = {
        Severity.ERROR: Fore.RED,
        Severity.WARNING: Fore.YELLOW,
        Severity.INFO: Fore.CYAN,
    }

    _SEV_LABEL = {
        Severity.ERROR: "ERROR",
        Severity.WARNING: "WARNING",
        Severity.INFO: "RECOMMEND",
    }

    _DIVIDER = "-" * shutil.get_terminal_size().columns

    def __init__(self, min_severity: Severity = Severity.INFO) -> None:
        self.min_severity = min_severity

    def run(self, suite: Dict[CheckGroup, List[Check]]) -> List[CheckRecord]:
        records: List[CheckRecord] = []

        for group, checks in suite.items():
            self._print_group_header(group)
            for check in checks:
                outcome = self._execute(check)
                record = CheckRecord(check=check, outcome=outcome)
                records.append(record)
                self._print_item(record)
            self._print_group_done(group)

        self._print_issues(records)
        return records

    @staticmethod
    def _execute(check: Check) -> CheckOutcome:
        try:
            return check.fn()
        except Exception as exc:
            return Failed(
                msg=str(exc),
                severity=Severity.ERROR,
                result_text="error",
                traceback=tb.format_exc(),
            )

    # ── rendering ─────────────────────────────────────────────────────────
    @staticmethod
    def _print_group_header(group: CheckGroup) -> None:
        print(f"- {Fore.CYAN}{group.title}{Style.RESET_ALL}")

    @staticmethod
    def _print_group_done(group: CheckGroup) -> None:
        print(f"- {Fore.CYAN}{group.title} - done{Style.RESET_ALL}")

    def _print_item(self, record: CheckRecord) -> None:
        print(f"    - {record.check.description} - {self._format_status(record)}")

    def _print_issues(self, records: List[CheckRecord]) -> None:
        issues = [
            r
            for r in records
            if r.failed and isinstance(r.outcome, Failed) and r.outcome.severity >= self.min_severity
        ]
        if not issues:
            return

        print()
        print(self._DIVIDER)
        print()

        for record in issues:
            outcome: Failed = record.outcome
            color = self._SEV_COLOR[outcome.severity]
            label = self._SEV_LABEL[outcome.severity]
            print(f"- {color}[{label}]{Style.RESET_ALL} {outcome.msg}")
            if outcome.traceback:
                print(f"    {Fore.RED}Traceback:{Style.RESET_ALL}")
                for line in outcome.traceback.splitlines():
                    print(f"        {line}")

    @staticmethod
    def _format_status(record: CheckRecord) -> str:
        outcome = record.outcome
        if isinstance(outcome, Skipped):
            return f"{Fore.YELLOW}skipped{Style.RESET_ALL} -- {Fore.LIGHTBLACK_EX}{outcome.reason}{Style.RESET_ALL}"
        if isinstance(outcome, Passed):
            return f"{Fore.GREEN}{outcome.result_text or 'ok'}{Style.RESET_ALL}"
        color = PrecheckRunner._SEV_COLOR[outcome.severity]
        return f"{color}{outcome.result_text or 'failed'}{Style.RESET_ALL}"
