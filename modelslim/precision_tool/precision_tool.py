# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import glob
import json
import os
import stat
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import torch_npu
import pandas as pd
from transformers import PreTrainedModel, AutoTokenizer
from tqdm import tqdm

from precision_tool.security import json_safe_load, json_safe_dump, get_valid_write_path
from precision_tool import logger

supported_dataset = ["boolq", "ceval_0_shot", "ceval_5_shot", "humaneval"]

supported_hardware = ["npu"]

TENSOR_TYPE_PYTORCH = "pt"
DATASET_HUMAN_EVAL = "humaneval"


def is_running_on_npu(device_name):
    return device_name.lower() == "npu"


class PrecisionTest:
    def __init__(self, model, tokenizer, dataset, batch_size, hardware_type,
                 tokenizer_return_type_id=False):
        """
        @param model:
            llm to run the test, should be an instance of transformers.PreTrainedModel
        @param dataset:
            dataset to test precision
        @param batch_size:
            batch_size to run inference
        @param hardware_type:
            currently only npu is supported
        @param tokenizer_return_type_id:
            tokenizer return token type id
        """
        self.logger = logger
        self.logger.info("Ready to init precision tool.")
        self.dataset = dataset
        if not self.__verify_dataset():
            raise TypeError("Dataset setting failed.")
        self.model = model
        if not self.__verify_model():
            raise TypeError("Invalid model.")
        self.device = model.device
        self.tokenizer_return_type_id = tokenizer_return_type_id
        if not isinstance(self.tokenizer_return_type_id, bool):
            raise TypeError("Tokenizer return type id must be bool.")
        self.tokenizer = tokenizer
        if not self.__verify_tokenizer():
            raise TypeError("Tokenizer must be matched with model.")
        self.batch_size = batch_size
        if not self.__verify_batch_size():
            raise TypeError("Batch size must be a positive integer.")
        self.hardware_type = hardware_type
        if self.hardware_type not in supported_hardware:
            raise TypeError(f"Hardware type {self.hardware_type} is not supported.")
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.script_path, "dataset", self.dataset)
        if not os.path.exists(self.dataset_path):
            raise EnvironmentError(f"Dataset was not found, valid path should be '{self.dataset_path}")
        self.result_file = ""
        self.logger.info("Precision test was inited.")

    def test(self):
        self.logger.info("Begin to run precision test.")
        torch.manual_seed(1)
        self.__run_test()
        self.logger.info("Precision test has finished.")

    def __verify_batch_size(self):
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            self.logger.error("Batch size must be an integer")
            return False
        if not self.batch_size > 0:
            self.logger.error("Batch size must be positive")
            return False
        return True

    def __verify_dataset(self):
        if not isinstance(self.dataset, str):
            self.logger.error("Dataset should be a string")
            return False
        if self.dataset not in supported_dataset:
            self.logger.error(f"Dataset {self.dataset} is not supported. Use {supported_dataset} instead.")
            return False
        if self.dataset == DATASET_HUMAN_EVAL:
            try:
                from human_eval.evaluation import evaluate_functional_correctness
            except Exception as e:
                self.logger.error(f"Import human_eval test package failed. Error info is as below. \n {e}")
                return False
        return True

    def __verify_model(self):
        if not isinstance(self.model, PreTrainedModel):
            self.logger.error(f"Model is not a valid model, type is {type(self.model)}")
            return False
        return True

    def __verify_tokenizer(self):
        text = "hello world"
        inputs = (self.tokenizer(text, return_tensors="pt", return_token_type_ids=self.tokenizer_return_type_id)
                  .to(self.model.device))
        try:
            with torch.no_grad():
                self.model(**inputs)
        except Exception as e:
            self.logger.error(f"{e}")
            self.logger.error("Model and tokenizer are not compatible. Cannot simply use model API.")
            self.logger.error(f"Model type is {type(self.model)}")
            self.logger.error(f"Tokenizer type is {type(self.tokenizer)}")
            return False
        try:
            with torch.no_grad():
                self.model.generate(**inputs, do_sample=False, max_new_tokens=1)
        except Exception as e:
            self.logger.error(f"{e}")
            self.logger.error("Model and tokenizer are not compatible. Cannot simply use generate API.")
            self.logger.error(f"Model type is {type(self.model)}")
            self.logger.error(f"Tokenizer type is {type(self.tokenizer)}")
            return False
        self.logger.info("Model and Tokenizer are compatible.")
        return True

    def __run_test(self):
        if self.dataset == 'ceval_0_shot':
            self.__run_full_dataset_ceval_0_shot()
        elif self.dataset == 'ceval_5_shot':
            self.__run_full_dataset_ceval_5_shot()
        elif self.dataset == 'boolq':
            self.__run_full_dataset_boolq()
        elif self.dataset == 'humaneval':
            self.__run_full_dataset_humaneval()
        else:
            raise TypeError("Unsupported dataset")

    def __run_full_dataset_ceval_0_shot(self):
        choices = ["A", "B", "C", "D"]
        choice_tokens = [self.tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]

        extraction_prompt = '综上所述，ABCD中正确的选项是：'

        def build_prompt(text):
            return "[Round {}]\n\n问：{}\n\n答：".format(1, text)

        def run_test():
            correct_total, sum_total = 0, 0
            for entry in glob.glob((Path(self.dataset_path) / "val/**/*.jsonl").as_posix(), recursive=True):
                correct, dataset = 0, []

                with open(entry, encoding='utf-8') as file:
                    for line in file:
                        single_json = json.loads(line)
                        dataset.append(single_json)

                dataset_num = len(dataset)

                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for batch in tqdm(dataloader):
                    texts = batch["inputs_pretokenized"]
                    queries = [build_prompt(query) for query in texts]
                    inputs = self.tokenizer(queries, padding=True, return_tensors=TENSOR_TYPE_PYTORCH, truncation=True,
                                            return_token_type_ids=self.tokenizer_return_type_id).to(self.device)
                    outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
                    intermediate_outputs = []
                    for idx in range(len(outputs)):
                        output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                        response = self.tokenizer.decode(output)
                        intermediate_outputs.append(response)
                    answer_texts = []
                    for text, intermediate in zip(texts, intermediate_outputs):
                        answer_texts.append(text + intermediate + "\n" + extraction_prompt)
                    input_tokens = []
                    for answer_text in answer_texts:
                        input_tokens.append(build_prompt(answer_text))
                    inputs = self.tokenizer(input_tokens, padding=True, return_tensors=TENSOR_TYPE_PYTORCH,
                                            truncation=True,
                                            return_token_type_ids=self.tokenizer_return_type_id).to(self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    logits = logits[:, choice_tokens]
                    preds = logits.argmax(dim=-1)
                    correct += (preds.cpu() == batch["label"]).sum().item()
                correct_total += correct
                sum_total += dataset_num
            return correct_total, sum_total

        with torch.no_grad():
            correct_total, sum_total = run_test()

        if sum_total == 0:
            self.logger.error("Did not ran any test, maybe wrong cevel_0_shot dataset folder.")
        else:
            self.logger.info(
                f"correct rate:{correct_total / sum_total}, correct total:{correct_total}, sum_total:{sum_total}")

    def __run_full_dataset_ceval_5_shot(self):
        choices = ["A", "B", "C", "D"]
        shot = 5

        def get_subject_mapping():
            subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
            with open(subject_mapping_path) as f:
                subject_mapping = json.load(f)
            return subject_mapping

        def load_csv_by_task_name(task_name, dataset_path):
            dev_file_path = get_valid_write_path(os.path.join(dataset_path, "dev", task_name + "_dev.csv"))
            dev_df = pd.read_csv(dev_file_path, header=None)[:shot + 1]
            val_file_path = get_valid_write_path(os.path.join(dataset_path, "val", task_name + "_val.csv"))
            val_df = pd.read_csv(val_file_path, header=None)

            dev_df = dev_df.iloc[1:, 1:]
            val_df = val_df.iloc[1:, 1:]
            return dev_df, val_df

        def format_subject(subject):
            str_units = subject.split("_")
            res = ""
            for entry in str_units:
                res += " " + entry
            return res

        def format_example(df, idx, include_answer=True):
            prompt = df.iloc[idx, 0]
            k = len(choices)
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
            prompt += "\nAnswer:"
            if include_answer:
                prompt += " {}\n\n".format(df.iloc[idx, k + 1])
            return prompt

        def gen_prompt(train_df, subject, k=-1):
            prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                format_subject(subject))
            if k == -1:
                k = train_df.shape[0]
            for i in range(k):
                prompt += format_example(train_df, i)
            return prompt

        correct_total = 0
        sum_total = 0

        subject_mapping = get_subject_mapping()
        index = 1
        for task_name in tqdm(subject_mapping):
            dev_df, val_df = load_csv_by_task_name(task_name, self.dataset_path)
            correct = 0
            task_len = val_df.shape[0]
            for i in range(math.ceil(task_len / self.batch_size)):
                q_num = self.batch_size if (i + 1) * self.batch_size <= task_len else task_len - i * self.batch_size
                prompt_ends = []
                for j in range(q_num):
                    prompt_ends.append(format_example(val_df, i * self.batch_size + j, include_answer=False))
                train_prompts = [gen_prompt(dev_df, task_name, shot)] * q_num
                prompt = [t + p for t, p in zip(train_prompts, prompt_ends)]
                labels = [val_df.iloc[i * self.batch_size + j, val_df.shape[1] - 1] for j in range(q_num)]
                prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt]
                inputs = self.tokenizer(prompts, padding=True, return_tensors=TENSOR_TYPE_PYTORCH, truncation=True,
                                        return_token_type_ids=self.tokenizer_return_type_id).to(self.device)
                outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20)
                answers = []
                for idx in range(len(outputs)):
                    output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                    response = self.tokenizer.decode(output)
                    answers.append(response)

                answer_results = [answer.lstrip()[0] if answer else "-1" for answer in answers]
                is_correct = []
                for answer_result, label in zip(answer_results, labels):
                    is_correct.append("Correct" if answer_result == label else "Wrong")
                correct += is_correct.count("Correct")
            correct_total += correct
            sum_total += task_len
            index += 1
        if sum_total == 0:
            self.logger.error(
                f"Did not ran any test, maybe wrong cevel_5_shot dataset folder. Current folder is {self.dataset_path}")
        else:
            self.logger.info(
                f"correct rate:{correct_total / sum_total}, correct total:{correct_total}, sum_total:{sum_total}")

    def __run_full_dataset_humaneval(self):
        def cleanup_code(code: str) -> str:
            code_splits = code.split("\n")
            is_empty_line = False
            ind_empty_line = None
            for i, line in enumerate(code_splits):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    is_empty_line = True
                    ind_empty_line = i
                    break
            if is_empty_line:
                code = "\n".join(code_splits[:ind_empty_line])
            else:
                end_words = ["\ndef", "\nclass", "\n#", "\nassert", '\n"""', "\nprint", "\nif", "\n\n\n"]
                for w in end_words:
                    if w in code:
                        code = code[:code.rfind(w)]
            return code

        def run_test():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(), recursive=True),
                              desc='global'):
                dataset = []
                with open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        dataset.append(line_json)

                samples = []
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for _, batch in enumerate(tqdm(dataloader)):
                    task_ids = [task_id.split('/')[1] for task_id in batch["task_id"]]
                    queries = [prompt.strip() for prompt in batch["prompt"]]
                    inputs = self.tokenizer(queries, padding=True, return_tensors=TENSOR_TYPE_PYTORCH,
                                            truncation=True).to(self.device)
                    outputs = self.model.generate(inputs=inputs.input_ids,
                                                  attention_mask=inputs.attention_mask, do_sample=False,
                                                  max_new_tokens=512)
                    for idx, output in enumerate(outputs.tolist()):
                        output = output[len(inputs["input_ids"][idx]):]
                        response = self.tokenizer.decode(output)
                        response_cleaned_up = cleanup_code(response)
                        result = dict(
                            task_id="HumanEval/" + task_ids[idx],
                            completion=response_cleaned_up,
                        )
                        samples += [result]
                self.__save_humaneval_res(samples)
                self.logger.info(f"Single question check result has been written to {self.result_file}.")
                result = evaluate_functional_correctness(self.result_file, [1], 4, 3.0, entry)
                self.logger.info(result)

        try:
            from human_eval.evaluation import evaluate_functional_correctness
        except ImportException as e:
            self.logger.error(f"Import human_eval test package failed. Error info is as below. \n {e}")
            return

        with torch.no_grad():
            try:
                run_test()
            except Exception as e:
                self.logger.error(f"Unable to run humaneval test. {e}")

    def __run_full_dataset_boolq(self):
        sample_yes = "How can we learning machine learning: yes"
        sample_no = "How can we learning machine learning: no"
        choice_tokens = [
            self.tokenizer([sample_yes], return_tensors=TENSOR_TYPE_PYTORCH, max_length=2048,
                           add_special_tokens=False).input_ids[0, -1].item(),
            self.tokenizer([sample_no], return_tensors=TENSOR_TYPE_PYTORCH, max_length=2048,
                           add_special_tokens=False).input_ids[0, -1].item()
        ]

        def build_prompt(title, text, passage):
            prompt = f"{title} -- {passage}\nQuestion: {text}?\nAnswer:"
            return prompt

        def run_test(choice_tokens, correct_total, sum_total):
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(), recursive=True),
                              desc='global'):
                dataset = []
                with open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        dataset.append(line_json)

                correct = 0
                dataset_num = len(dataset)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for _, batch in enumerate(tqdm(dataloader)):
                    titles = batch["title"]
                    texts = batch["question"]
                    passages = batch["passage"]
                    queries = []
                    for title, query, passage in zip(titles, texts, passages):
                        queries.append(build_prompt(title, query, passage))
                    inputs = self.tokenizer(queries, padding=True, return_tensors=TENSOR_TYPE_PYTORCH, truncation=True,
                                            return_token_type_ids=self.tokenizer_return_type_id).to(self.model.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    logits_softmax = F.log_softmax(logits.float(), dim=-1)

                    logits_softmax = logits_softmax[:, choice_tokens]
                    for idx, ans in enumerate(batch['answer']):
                        choice = (logits_softmax[idx, 0] > logits_softmax[idx, 1]).cpu()
                        correct += 1 if choice == ans else 0
                correct_total += correct
                sum_total += dataset_num
            return correct_total, sum_total

        correct_total = 0
        sum_total = 0
        with torch.no_grad():
            correct_total, sum_total = run_test(choice_tokens, correct_total, sum_total)
        if sum_total == 0:
            self.logger.error("Did not ran any test, maybe wrong humaneval dataset folder.")
        else:
            self.logger.info(
                f"correct rate:{correct_total / sum_total}, correct total:{correct_total}, sum_total:{sum_total}")

    def __save_humaneval_res(self, results):
        self.result_file = os.path.dirname(os.path.abspath(__file__)) + os.sep + "result.jsonl"
        mode = stat.S_IWUSR | stat.S_IRUSR
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(self.result_file, flags=flags, mode=mode), "w") as fp:
            for result in results:
                fp.write((json.dumps(result) + "\n"))