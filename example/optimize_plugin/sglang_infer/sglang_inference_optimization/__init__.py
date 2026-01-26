def register():
    from sglang_inference_optimization.simulator import SGLangSimulator
    from sglang_inference_optimization.settings import CusSettings
    from msserviceprofiler.modelevalstate.optimizer.register import register_simulator
    from msserviceprofiler.modelevalstate.config.config import register_settings
    register_simulator("sgl_infer", SGLangSimulator)
    register_settings(lambda : CusSettings())