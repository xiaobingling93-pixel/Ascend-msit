import os
from mindformers import MindFormerConfig, LlamaConfig, ChatGLM2Config, \
                        init_context, TransformerOpParallelConfig
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import RoundToNearest as RTN

class QuantConfig:
    def __init__(self, config_path, 
                       modle_name,
                       framework,
                       mode=PTQMode.QUANTIZE,
                       backend=BackendTarget.ASCEND):
        if not isinstance(config_path, str):
            raise ValueError("config_path must be a string")
        if not isinstance(modle_name, str):
            raise ValueError("modle_name must be a string")
        if not os.path.exists(config_path):
            raise FileNotFoundError("config file not found")

        if framework == "ms":
            self.config_path = config_path
            self.model_name = modle_name
            self.mode = mode
            self.backend = backend
            self.PTQcfg = PTQConfig(mode=mode, backend=backend)
            self.cfg = self.create_mfconfig(self.config_path)
        elif framework == "torch":
            raise NotImplementedError("torch framework is not supported yet")
        else:
            raise ValueError("framework must be 'ms' or 'torch'")

    def create_mfconfig(self, config_path):
        """Create mindformers config for llama2 network for example."""
        config = MindFormerConfig(config_path)
        if self.model_name.startswith("llama"):
            config.model.model_config = LlamaConfig(**config.model.model_config)
        elif self.model_name.startswith("chatglm"):
            config.model.model_config = ChatGLM2Config(**config.model.model_config)

        init_context(use_parallel=config.use_parallel, context_config=config.context, parallel_config=config.parallel)

        parallel_config = TransformerOpParallelConfig(**config.parallel_config)
        config.model.model_config.parallel_config = parallel_config
        return config
