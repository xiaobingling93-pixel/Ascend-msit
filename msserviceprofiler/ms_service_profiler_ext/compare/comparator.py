import os
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from collector import FileCollector

class ComparatorFactory:
    _registry = {}

    @classmethod
    def register(cls, file_name: str):
        def decorator(subclass):
            cls._registry[file_name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, path_a: str, path_b: str):
        base_name_a = os.path.basename(path_a).lower()
        base_name_b = os.path.basename(path_b).lower()
        
        if base_name_a != base_name_b:
            raise ValueError("File names mismatch")
        
        if base_name_a not in cls._registry:
            raise ValueError(f"Unsupported file type: {base_name_a}")

        return cls._registry[base_name_a](path_a, path_b)

@dataclass
class MetricAnalysis:
    """增强的指标分析结果"""
    metric_name: str
    absolute_changes: Dict[str, float]
    relative_changes: Dict[str, float]
    top_changes: List[Tuple[str, float, float]]  # (字段, 绝对变化, 相对变化)
    needs_attention: bool = False
    advice: Optional[str] = None

class EnhancedReportGenerator:
    """增强的报告生成器"""
    COLOR_RED = '\033[91m'
    COLOR_GREEN = '\033[92m'
    COLOR_YELLOW = '\033[93m'
    COLOR_RESET = '\033[0m'
    UP_ARROW = '\u2191'
    DOWN_ARROW = '\u2193'

    ANALYSIS_RULES = {
        'Long Tail': {
            'threshold': 0.3,
            'advice': "P99与P90差距扩大{value:.1f}%，建议优化高百分位性能"
        },
        'Range': {
            'threshold': 0.2,
            'advice': "数值范围变化{value:.1f}%，请检查极端值"
        },
        'Average': {
            'threshold': 0.1,
            'advice': "平均值变化{value:.1f}%，需关注整体趋势"
        }
    }

    def generate(self, analysis_results: List[MetricAnalysis], total_score: float):
        """生成优化后的报告"""
        self._print_header()
        attention_metrics = []
        
        # 第一阶段：显示所有指标概览
        for result in analysis_results:
            self._print_metric_overview(result)
            if result.needs_attention:
                attention_metrics.append(result)

        # 第二阶段：显示需要关注的建议
        if attention_metrics:
            self._print_advice_section(attention_metrics)

        # 第三阶段：显示综合评分
        self._print_final_score(total_score)

    def _print_header(self):
        """打印报告头部"""
        print("\n" + "="*60)
        print(f"{'性能分析报告（优化版）':^60}")
        print("="*60 + "\n")

    def _print_metric_overview(self, analysis: MetricAnalysis):
        """打印指标概览"""
        print(f"[ {analysis.metric_name} ]")
        print("变化最显著的三个指标：")
        
        for field, abs_change, rel_change in analysis.top_changes[:3]:
            abs_str = self._format_value(abs_change, is_percent=False)
            rel_str = self._format_value(rel_change, is_percent=True)
            print(f"  {field:<15}: {abs_str} | {rel_str}")

        print("-"*60)

    def _print_advice_section(self, metrics: List[MetricAnalysis]):
        """打印建议部分"""
        print("\n" + "="*60)
        print(f"{'关键改进建议':^60}")
        print("="*60)
        
        for metric in metrics:
            print(f"\n[ {metric.metric_name} ]")
            print(f"  {self.COLOR_YELLOW}{metric.advice}{self.COLOR_RESET}")

    def _print_final_score(self, score: float):
        """打印最终评分"""
        color = self.COLOR_RED if score > 0 else self.COLOR_GREEN
        symbol = self.UP_ARROW if score > 0 else self.DOWN_ARROW
        advice = self._get_score_advice(score)
        
        print("\n" + "="*60)
        print(f"{'综合评估':^60}")
        print("="*60)
        print(f"评分变化: {color}{score:+.2f}{symbol}{self.COLOR_RESET}")
        print(f"改进建议: {advice}")
        print("="*60)

    def _format_value(self, value: float, is_percent: bool) -> str:
        """带格式的数值显示"""
        template = "{:+.1f}%" if is_percent else "{:+.2f}"
        value_str = template.format(value*100 if is_percent else value)
        
        if value > 0:
            return f"{self.COLOR_RED}{value_str}{self.UP_ARROW}{self.COLOR_RESET}"
        if value < 0:
            return f"{self.COLOR_GREEN}{value_str}{self.DOWN_ARROW}{self.COLOR_RESET}"
        return value_str

    def _get_score_advice(self, score: float) -> str:
        """根据评分获取建议"""
        if score > 1:
            return "系统性能显著下降，需要立即优化！"
        if score > 0.5:
            return "存在可观测的性能下降，建议优先处理关键指标"
        if score < -1:
            return "系统性能有重大提升，请确认变更内容"
        if score < -0.5:
            return "性能有所改善，继续保持优化方向"
        return "性能波动在正常范围内"
    

class BaseComparator(ABC):
    def __init__(self, path_a: str, path_b: str):
        self.path_a = path_a
        self.path_b = path_b
    
    @abstractmethod
    def compare(self) -> List[MetricAnalysis]:
        pass


@ComparatorFactory.register('batch_summary.csv')
@ComparatorFactory.register('request_summary.csv')
@ComparatorFactory.register('service_summary.csv')
class EnhancedCSVComparator(BaseComparator):
    METRIC_WEIGHTS = {
        'average': 0.4,
        'max': 0.2,
        'min': 0.1,
        'P50': 0.1,
        'P90': 0.1,
        'P99': 0.1
    }

    def __init__(self, path_a, path_b):
        super().__init__(path_a, path_b)
        self.data_a = self._parse_csv(path_a)
        self.data_b = self._parse_csv(path_b)
        self.report_generator = EnhancedReportGenerator()

    def _parse_csv(self, path):
        """解析CSV文件"""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            return {
                row[0]: {h: self._try_float(v) for h, v in zip(headers, row)}
                for row in reader if row
            }

    def _try_float(self, value):
        """安全类型转换"""
        try:
            return float(value)
        except ValueError:
            return value

    def compare(self) -> List[MetricAnalysis]:
        """执行比较分析"""
        analysis_results = []
        total_score = 0.0

        all_metrics = set(self.data_a.keys()) | set(self.data_b.keys())
        for metric in sorted(all_metrics):
            analysis = self._analyze_metric(metric)
            analysis_results.append(analysis)
            total_score += sum(
                abs_change * self.METRIC_WEIGHTS[field]
                for field, abs_change in analysis.absolute_changes.items()
            )

        self.report_generator.generate(analysis_results, total_score)
        return analysis_results

    def _analyze_metric(self, metric: str) -> MetricAnalysis:
        """分析单个指标"""
        a_data = self.data_a.get(metric, {})
        b_data = self.data_b.get(metric, {})

        # 计算绝对变化和相对变化
        abs_changes = {}
        rel_changes = {}
        
        for field in self.METRIC_WEIGHTS:
            a_val = a_data.get(field, 0)
            b_val = b_data.get(field, 0)
            
            abs_change = b_val - a_val
            try:
                rel_change = abs_change / a_val if a_val != 0 else 0
            except ZeroDivisionError:
                rel_change = 0
                
            abs_changes[field] = abs_change
            rel_changes[field] = rel_change

        # 获取变化最大的三个指标
        sorted_changes = sorted(
            abs_changes.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_changes = [
            (field, abs_changes[field], rel_changes[field])
            for field, _ in sorted_changes
        ]

        # 生成建议
        advice, needs_attention = self._generate_advice(
            metric, abs_changes, rel_changes
        )

        return MetricAnalysis(
            metric_name=metric,
            absolute_changes=abs_changes,
            relative_changes=rel_changes,
            top_changes=top_changes,
            needs_attention=needs_attention,
            advice=advice
        )

    def _generate_advice(self, metric: str, abs_changes: Dict, rel_changes: Dict) -> Tuple[str, bool]:
        """生成改进建议"""
        advice = []
        needs_attention = False

        # 长尾分析
        p99_change = rel_changes.get('P99', 0)
        p90_change = rel_changes.get('P90', 0)
        long_tail = p99_change - p90_change
        if abs(long_tail) > 0.3:
            advice.append(
                f"长尾差异变化 {long_tail*100:+.1f}%，建议关注P99/P90分布"
            )
            needs_attention = True

        # 范围分析
        range_change = rel_changes.get('max', 0) - rel_changes.get('min', 0)
        if abs(range_change) > 0.2:
            advice.append(
                f"数值范围变化 {range_change*100:+.1f}%，请检查极端值"
            )
            needs_attention = True

        # 平均值分析
        avg_change = rel_changes.get('average', 0)
        if abs(avg_change) > 0.1:
            advice.append(
                f"平均值变化 {avg_change*100:+.1f}%，需关注整体趋势"
            )
            needs_attention = True

        return "; ".join(advice), needs_attention


class ComparisonExecutor(object):
    def __init__(self, file_collector: FileCollector, max_workers: int=1):
        self.file_collector = file_collector
        self.max_workers = max_workers

    def submit(self, dir_path_a: str, dir_path_b: str):
        for path_a, path_b in self.file_collector.collect_pairs(dir_path_a, dir_path_b):
            comparator = ComparatorFactory.create(path_a, path_b)
            comparator.compare()