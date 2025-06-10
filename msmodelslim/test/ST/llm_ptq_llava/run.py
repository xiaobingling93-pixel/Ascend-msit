from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests
import os 

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig


LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llava-v15-7b-hf/"
processor = AutoProcessor.from_pretrained(LOAD_PATH, pad_token="<pad>")
 
model = LlavaForConditionalGeneration.from_pretrained(
    LOAD_PATH, 
    torch_dtype=torch.float16, 
    device_map='auto'
).eval()
 
 
images_list = os.listdir(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/")
image = Image.open(os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", images_list[0]))
prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
calib_data = []
for i in images_list[:1]:
    image = Image.open(os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", i))
    data_1 = processor(images=image, text=prompt, return_tensors="pt").to('npu')
    calib_data.append([data_1.data['input_ids'], data_1.data['pixel_values'], data_1.data['attention_mask']])
 
 
disable_names = []
 
anti_config = AntiOutlierConfig(
    a_bit=8,
    w_bit=8,
    anti_method="m2",
    dev_type="npu",
    dev_id=model.device.index,
    )
anti_outlier = AntiOutlier(model, calib_data=calib_data , cfg=anti_config)
anti_outlier.process()
 
 
quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=2,
    mm_tensor=False
)
 
calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_llava")