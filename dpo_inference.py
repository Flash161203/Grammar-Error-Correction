from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
#### Merge SFT adapter
sft_adapter = "AY2324S2-CS4248-Team-47/StableLM-WI_Locness"
peft_config = PeftConfig.from_pretrained(sft_adapter)
model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True, attn_implementation="flash_attention_2") 
    
tokenizer = AutoTokenizer.from_pretrained(
    sft_adapter, use_fast=True
)

model.resize_token_embeddings(len(tokenizer))
        
model = PeftModel.from_pretrained(model, sft_adapter) 
model = model.merge_and_unload(safe_merge=True)

### Merge DPO adapter
dpo_adapter = "AY2324S2-CS4248-Team-47/StableLM-DPO-Ultrafeedback"
peft_config = PeftConfig.from_pretrained(dpo_adapter)
model = PeftModel.from_pretrained(model, dpo_adapter)
model = model.merge_and_unload(safe_merge=True)
generator = pipeline("text-generation", model=model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer)

INSTRUCTION = "Rewrite the given text and correct grammar, spelling, and punctuation errors."
def correct_text(original_text: str, k: int=1, apply_template=True, batched=False, bsize: int=2):
    if apply_template:
        if batched:
            original_text = [text.strip(" \n\t") for text in original_text]
            prompt = [f'### Instruction:\n{INSTRUCTION}\n### Input:\n{text}{tokenizer.eos_token}\n### Response:\n' for text in original_text]
        else:
            prompt = f'### Instruction:\n{INSTRUCTION}\n### Input:\n{original_text}{tokenizer.eos_token}\n### Response:\n'
    else:
        prompt = original_text
    # print(f'prompt: {prompt}')
    if batched:
        outputs = generator(prompt, do_sample=True, max_length=1024, batch_size=bsize, num_return_sequences=k, return_full_text=False, clean_up_tokenization_spaces=True)
    else:
       outputs = generator(prompt, do_sample=True, max_length=1024, num_return_sequences=k, return_full_text=False, clean_up_tokenization_spaces=True) 
    return outputs

if __name__ == "__main__":
    while True:
        text = input().strip()
        for output in correct_text(text, 1):
            print(output)
            print('-----------')