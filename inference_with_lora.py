from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
import wandb

def download_artifact_from_wandb():
    artifact_path = "ay2324s2-cs4248-team-47/finetune-pretrained-transformer/model-mhpecfs6:v0"
    api = wandb.Api()
    artifact = api.artifact(artifact_path)

    artifact_dir = artifact.download()

ckpt_path = "./artifacts/model-mhpecfs6:v0"
config = PeftConfig.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]'})
    model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, ckpt_path)
model = model.merge_and_unload(safe_merge=True)

generator = pipeline("text-generation", model=model, device_map="auto", torch_dtype=torch.float16, tokenizer=tokenizer)
INSTRUCTION = "Rewrite the given text and correct grammar, spelling, and punctuation errors."
def correct_text(original_text: str, k: int=1):
    prompt = f'### Instruction:\n{INSTRUCTION}\n### Input:\n{original_text}\n### Response:\n'
    print(f'prompt: {prompt}')
    len_tokenized_input = len(tokenizer(prompt)["input_ids"])
    outputs = generator(prompt, do_sample=True, max_new_tokens=1024-len_tokenized_input, num_return_sequences=k, return_full_text=False)
    
    return outputs

if __name__ == "__main__":
    while True:
        text = input().strip()
        for output in correct_text(text, 5):
            print(output)
            print('-----------')