# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "optimum",
#   "optimum[onnxruntime]",
# ]
# ///

from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

model_id = "gpt2"
model = ORTModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)
model.save_pretrained("../models/base")