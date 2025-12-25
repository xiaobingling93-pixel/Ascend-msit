def register():
    from vllm_inference_optimization.benchmark import VllmBenchMark
    from vllm_inference_optimization.simulator import VllmSimulator
    from vllm_inference_optimization.settings import CusSettings
    from msserviceprofiler.modelevalstate.optimizer.register import register_simulator, register_benchmarks
    from msserviceprofiler.modelevalstate.config.config import register_settings
    register_benchmarks("vllm_infer_benchmark", VllmBenchMark)
    register_simulator("vllm_infer", VllmSimulator)
    register_settings(lambda : CusSettings())