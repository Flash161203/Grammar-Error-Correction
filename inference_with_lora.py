from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

ckpt_path = "./models/checkpoint-375"
config = PeftConfig.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, ckpt_path)
model = model.merge_and_unload(safe_merge=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
generator = pipeline("text-generation", model=model, device_map="auto", tokenizer=tokenizer)
INSTRUCTION = "Rewrite the given text and correct grammar, spelling, and punctuation errors."

original_text = input("Enter text to correct: ")
prompt = f'### Instruction:\n{INSTRUCTION}\n### Input:\n{original_text}\n### Response:\n'
len_tokenized_input = len(tokenizer(prompt)["input_ids"])
outputs = generator(prompt, do_sample=True, max_new_tokens= len_tokenized_input, num_beams=5, num_return_sequences=1, return_full_text=False)
print(outputs[0])