from pathlib import Path
import torch
import torch_npu
from msit_llm.transform.torch_to_atb_python import ATBModel
from safetensors.torch import safe_open
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from atb_model_symbol import Model


def load_model_dict(model_path):
    if Path(model_path).is_file():
        state_dict = torch.load(model_path)
        return state_dict
    elif Path(model_path).is_dir():
        suffix_list = ['.bin', '.safetensors', '.pt']
        for suffix in suffix_list:
            file_list = list(Path(model_path).glob('*' + suffix))
            if not file_list:
                continue
            state_dict = {}
            for fp in file_list:
                if suffix == '.safetensors':
                    with safe_open(fp, framework='pt') as ff:
                        ss = {kk: ff.get_tensor(kk).half() for kk in ff.keys()}
                else:
                    ss = torch.load(fp)
                state_dict.update(ss)
            return state_dict
    return {}


class CausalLM(PreTrainedModel):
    def __init__(self, model_path):
        config = AutoConfig.from_pretrained(model_path)
        super().__init__(config)

        device = torch.device(f'npu')
        torch.npu.set_device(device)        
        self.placeholder = torch.zeros(1, device=device)

        self.atb_model = ATBModel(Model())
        weights = load_model_dict(model_path)
        self.atb_model.set_weights(weights)

    def init_kv_cache(self):
        self.atb_model.init_kv_cache()

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs    

    def forward(self, input_ids, position_ids, use_cache=False, *args, **kwargs):
        if input_ids.dim() == 2:
            input_ids = input_ids[0]

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[0]).to(input_ids)

        if position_ids.dim() == 2:
            position_ids = position_ids[0]

        if use_cache:
            past_key_values = (self.placeholder, self.placeholder)
        else:
            past_key_values = None
            self.atb_model.init_kv_cache()

        out = self.atb_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        logits = out['output'].unsqueeze(0)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values
            )    


class Runner:
    def __init__(self, model_path):
        self.model = CausalLM(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.max_input_length = 20
        self.max_output_length = 20
        self.batch_size = 1
    
    def warm_up(self):
        dummy_input_ids_full = torch.randint(
            0, 32000, [self.batch_size, self.max_input_length], dtype=torch.long).npu()
        self.model.generate(inputs=dummy_input_ids_full, do_sample=False, max_new_tokens=10)

    def infer(self, input_text, use_cache=True):
        if isinstance(input_text, str):
            input_text = [input_text] * self.batch_size

        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        self.model.init_kv_cache()

        # Prefill
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs=inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                max_new_tokens=1
            )

        # Deocde
        with torch.no_grad():
            generate_ids = self.model.generate(
                inputs=inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                max_new_tokens=self.max_output_length,
                use_cache=use_cache
            )

        generate_text = self.tokenizer.batch_decode(
            generate_ids[:, :], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        output_text = generate_text[0][:]
        return output_text
    

def main():
    runner = Runner("model_path_symbol")

    input_text = '好雨知时节，当春'

    runner.warm_up()
    output_text = runner.infer(input_text, use_cache=False)

    import logging
    logger = logging.getLogger()
    logger.info('-' * 40)
    logger.info('Input:%s', input_text)
    logger.info('Output:%s', output_text)


if __name__ == '__main__':
    main()



