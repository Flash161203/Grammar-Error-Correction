# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
import sys
from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, training_args, model_config = parser.parse_json_file(sys.argv[1])
    else:
        args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    ################
    # Dataset
    ################
    import random

    def formatting_prompts_func(examples):
        # format dataset following https://github.com/huggingface/trl/pull/444#issue-1760952763
        # prompt from https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset/viewer/default/test?q=correct&row=77
        INSTRUCTION = "Rewrite the given text and correct grammar, spelling, and punctuation errors."
        output_texts = []
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

            text = f"### Instruction:\n{INSTRUCTION}\n### Input:\n{original_text}\n### Response:\n{corrected_text}"

            output_texts.append(text)
        return output_texts

    raw_datasets = load_dataset(args.dataset_name, args.config)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    ################
    # Optional rich context managers
    ###############
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the SFTTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    ################
    # Training
    ################
    import wandb
    os.environ["WANDB_LOG_MODEL"] = "end"
    run = wandb.init(entity="ay2324s2-cs4248-team-47", project="finetune-pretrained-transformer")
    peft_config = get_peft_config(model_config)
    peft_config.use_rslora = True
    
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
    wandb.finish()
