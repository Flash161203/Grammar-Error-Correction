from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
ckpt_path = "./models/checkpoint-376"
config = PeftConfig.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, ckpt_path)
model = model.merge_and_unload(safe_merge=True)

generator = pipeline("text-generation", model=model, device_map="auto", torch_dtype=torch.float16, tokenizer=tokenizer)
INSTRUCTION = "Rewrite the given text and correct grammar, spelling, and punctuation errors."
def correct_text(original_text: str):
    prompt = f'### Instruction:\n{INSTRUCTION}\n### Input:\n{original_text}\n### Response:\n'
    print(f'prompt: {prompt}')
    len_tokenized_input = len(tokenizer(prompt)["input_ids"])
    outputs = generator(prompt, do_sample=True, max_new_tokens= len_tokenized_input, top_p=0.95, num_return_sequences=1, return_full_text=False)
    
    return outputs[0]