import subprocess
from typing import Optional, Tuple
from loguru import logger
from msserviceprofiler.modelevalstate.config.config import get_settings, OptimizerConfigField, VllmConfig
from msserviceprofiler.modelevalstate.config.custom_command import VllmCommand
from msserviceprofiler.modelevalstate.optimizer.interfaces.simulator import SimulatorInterface


class VllmSimulator(SimulatorInterface):
    def __init__(self, config: Optional[VllmConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            if settings.name != "vllm-inference-optimization":
                raise ValueError("Settings is invalidator.")
            self.config = settings.vllm
        super().__init__(*args, process_name=self.config.process_name, **kwargs)

        self.command = VllmCommand(self.config.command).command

    @property
    def base_url(self) -> str:
        """
        获取服务的base url 属性
        Returns:

        """
        logger.info(f"http://127.0.0.1:{self.config.command.port}/health")
        return f"127.0.0.1:{self.config.command.port}/health"

    def update_command(self):
        self.command = VllmCommand(self.config.command).command

    def stop(self, del_log: bool = True):
        """
        运行时，其他的准备工作。
        Returns:

        """
        try:
            subprocess.run(["pkill", "-15", "vllm"], stderr=subprocess.STDOUT, text=True)
        except subprocess.SubprocessError:
            pass
        super().stop(del_log)
