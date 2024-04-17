import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel, PeftConfig
import json
from huggingface_hub import login
from evaluate import load
import wandb


def login_to_wandb():
    wandb.login(key="6aa53325095dd9aded5a7ff48de5c3808e6aa7f9")


def login_to_hf_hub():
    login(token="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")


def process_split(dataset_name, dataset_config) -> DatasetDict:
    raw_datasets = load_dataset(dataset_name, dataset_config)
    train_test_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=0)
    raw_datasets["train"] = train_test_split["train"]
    raw_datasets["test"] = train_test_split["test"]
    return raw_datasets


def get_prompt(text, tokenizer):
    INSTRUCTION = "Correct spelling, punctuation and grammatical errors in this text:"
    prompt = f"{INSTRUCTION}{text}"
    return prompt


def preprocess_pretrained_model(examples):
    model_checkpoint = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, eos_token="")
    inputs = examples["text"]
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, return_offsets_mapping=True
    )

    labels_out = []
    offset_mapping = model_inputs.pop("offset_mapping")
    for i in range(len(model_inputs["input_ids"])):
        example_idx = i

        start_idx = offset_mapping[i][0][0]
        end_idx = offset_mapping[i][-2][1]

        edits = examples["edits"][example_idx]

        corrected_text = inputs[example_idx][start_idx:end_idx]

        for start, end, correction in reversed(
            list(zip(edits["start"], edits["end"], edits["text"]))
        ):
            if start < start_idx or end > end_idx:
                continue
            start_offset = start - start_idx
            end_offset = end - start_idx
            if correction == None:
                correction = tokenizer.unk_token
            corrected_text = (
                corrected_text[:start_offset] + correction + corrected_text[end_offset:]
            )

        labels_out.append(corrected_text)

    prompts = [get_prompt(text, tokenizer) for text in examples["text"]]
    return {"prompts": prompts, "labels": labels_out, "inputs": examples["text"]}


def run_pipeline_and_evaluate():
    raw_dataset = process_split("wi_locness", "wi")

    artifact_path = (
        "ay2324s2-cs4248-team-47/finetune-pretrained-transformer/flant5-lora:v1"
    )
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    config = PeftConfig.from_pretrained(artifact_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path, load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, artifact_dir)
    model = model.merge_and_unload(safe_merge=True)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"model loaded: {artifact_path}")
    remove_columns = [
        col for col in raw_dataset["train"].column_names if col not in ("cefr", "id")
    ]

    test_dataset = raw_dataset.map(
        preprocess_pretrained_model,
        batched=True,
        remove_columns=remove_columns,
    )["test"]

    test_dataset = test_dataset[:2]

    test_prompts = test_dataset["prompts"]
    test_inputs = test_dataset["inputs"]
    test_references = test_dataset["labels"]
    nested_test_references = [[ref] for ref in test_references]
    test_ids = test_dataset["id"]
    test_cefr = test_dataset["cefr"]

    predictions = pipe(
        test_prompts,
        do_sample=True,
        max_new_tokens=512,
        num_return_sequences=1,
    )

    print(predictions)
    predictions = [pred["generated_text"] for pred in predictions]

    print("Preds and refs ready!")

    bleu = load("bleu")
    rouge = load("rouge")
    bertscore = load("bertscore")

    bleu_score = bleu.compute(
        predictions=predictions, references=nested_test_references
    )
    rouge_score = rouge.compute(predictions=predictions, references=test_references)
    bert_score = bertscore.compute(
        predictions=predictions, references=test_references, lang="en"
    )

    print(bleu_score)
    print(rouge_score)
    print(bert_score)

    print(f"\nInput text: \n{test_inputs[0]}\n")
    print(f"Reference corrected text: \n{test_references[0]}\n")
    print(f"Model output: \n{predictions[0]}\n")

    data = {
        "inputs": test_inputs,
        "predictions": predictions,
        "references": test_references,
        "ids": test_ids,
        "cefr": test_cefr,
    }
    with open("t5_base_data.json", "w") as f:
        json.dump(data, f)

    metrics = {"bleu": bleu_score, "rouge": rouge_score, "bertscore": bert_score}
    with open("t5_base_metrics.txt", "w") as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    login_to_wandb()
    login_to_hf_hub()
    run_pipeline_and_evaluate()
