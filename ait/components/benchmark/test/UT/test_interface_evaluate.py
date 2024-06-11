from ais_bench.evaluate.interface import Evaluator


class TestClass:
    @staticmethod
    def fake_generate_func(prompt):
        return "A"

    def test_ceval(self):
        evaluator = Evaluator(self.fake_generate_func, "ceval", shot=5)
        evaluator.evaluate()
        evaluator.evaluate("edit-distance")

    def test_mmlu(self):
        evaluator = Evaluator(self.fake_generate_func, "mmlu", shot=5)
        evaluator.evaluate()
        evaluator.evaluate("edit-distance")

    def test_gsm8k(self):
        evaluator = Evaluator(self.fake_generate_func, "gsm8k", shot=5)
        evaluator.evaluate()
        evaluator.evaluate("edit-distance")
