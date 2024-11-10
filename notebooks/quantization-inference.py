from optimum.quanto import QuantizedModelForCausalLM
import torch
from llmlib.minicpm import MiniCPM
from transformers import AutoProcessor

modeldir = "MiniCPM-V-2_6-int4"
qmodel = QuantizedModelForCausalLM.from_pretrained(
    modeldir, trust_remote_code=True, dtype=torch.bfloat16
)

qmodel.to("cuda")
qmodel.eval()
qmodel._wrapped.processor = AutoProcessor.from_pretrained(
    "openbmb/MiniCPM-V-2_6", trust_remote_code=True
)

llm = MiniCPM(model=qmodel)

llm.chat(prompt="What is the meaning of life?")
