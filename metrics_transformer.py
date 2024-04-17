import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel, PeftConfig
import json
from huggingface_hub import login
from evaluate import load


def login_to_hf_hub():
    login(token="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")


def process_split(dataset_name, dataset_config) -> DatasetDict:
    raw_datasets = load_dataset(dataset_name, dataset_config)
    train_test_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=0)
    raw_datasets["train"] = train_test_split["train"]
    raw_datasets["test"] = train_test_split["test"]
    return raw_datasets


def preprocess_baseline_transformer(dataset):
    model_checkpoint = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, eos_token=""
    )  # doesnt actually set eos to "" , so using max_length=511 for now
    inputs = dataset["text"]
    model_inputs = tokenizer(
        inputs, max_length=511, truncation=True, return_offsets_mapping=True
    )

    labels_out = []
    offset_mapping = model_inputs.pop("offset_mapping")
    for i in range(len(model_inputs["input_ids"])):
        dataset_idx = i
        start_idx = offset_mapping[i][0][0]
        end_idx = offset_mapping[i][-2][1]
        edits = dataset["edits"][dataset_idx]
        corrected_text = inputs[dataset_idx][start_idx:end_idx]

        for start, end, correction in reversed(
            list(zip(edits["start"], edits["end"], edits["text"]))
        ):
            if start < start_idx or end > end_idx:
                continue
            start_offset = start - start_idx
            end_offset = end - start_idx
            if correction is None:
                correction = tokenizer.unk_token
            corrected_text = (
                corrected_text[:start_offset] + correction + corrected_text[end_offset:]
            )
        labels_out.append(corrected_text)

    labels_out = tokenizer(labels_out, max_length=512, truncation=True)
    model_inputs["labels"] = labels_out["input_ids"]
    return model_inputs


def run_pipeline_and_evaluate():
    raw_dataset = process_split("wi_locness", "wi")

    model_path = (
        "AY2324S2-CS4248-Team-47/GEC-Baseline-Transformer"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    test_dataset = raw_dataset.map(
        preprocess_baseline_transformer,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )["test"]

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"model loaded: {model_path}")

    test_dataset = test_dataset

    test_inputs = tokenizer.batch_decode(test_dataset["input_ids"])
    predictions = pipe(test_inputs, do_sample=True, max_new_tokens=512, batch_size=16)

    predictions = [pred["generated_text"] for pred in predictions]

    test_references = tokenizer.batch_decode(test_dataset["labels"])
    nested_test_references = [[ref] for ref in test_references]

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

    print(f"\nInput text: \n{test_inputs[1]}\n")
    print(f"Reference corrected text: \n{test_references[1]}\n")
    print(f"Model output: \n{predictions[1]}\n")

    data = {
        "inputs": test_inputs,
        "predictions": predictions,
        "references": test_references,
    }
    with open("data.json", "w") as f:
        json.dump(data, f)

    metrics = {"bleu": bleu_score, "rouge": rouge_score, "bertscore": bert_score}
    with open("metrics.txt", "w") as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    login_to_hf_hub()
    run_pipeline_and_evaluate()
