from datasets import load_dataset, DatasetDict
dataset_name = "wi_locness"
dataset_config = "wi"
test_size = 0.1
random_seed = 0

def get_raw_splits() -> DatasetDict:
    raw_datasets = load_dataset(dataset_name, dataset_config)
    train_test_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=0)
    raw_datasets["train"] = train_test_split["train"]
    raw_datasets["test"] = train_test_split["test"]
    return raw_datasets

def apply_prompt_template(examples, tokenizer) -> dict[str, list[str]]:
    # format dataset following https://github.com/huggingface/trl/pull/444#issue-1760952763
    # prompt from https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset/viewer/default/test?q=correct&row=77
    INSTRUCTION = "Rewrite the given text and correct grammar, spelling, and punctuation errors."
    prompts = []
    for i in range(len(examples["text"])):
        edits = examples["edits"][i]
        original_text = examples["text"][i]
        corrected_text = examples["text"][i]
        for start, end, correction in reversed(
            list(
                zip(
                    edits["start"],
                    edits["end"],
                    edits["text"],
                )
            )
        ):
            if correction == None:
                correction = tokenizer.unk_token  # what to do with None?
            corrected_text = (
                corrected_text[:start] + correction + corrected_text[end:]
            )

        prompt = (
            f'### Instruction:\n'
            f'{INSTRUCTION}\n'
            f'### Input:\n{original_text}{tokenizer.eos_token}\n'
            '### Response:\n'
            f'{corrected_text}{tokenizer.eos_token}'
        )
        prompts.append(prompt)
    return {'text': prompts}