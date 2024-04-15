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
import multiprocessing
import os
from contextlib import nullcontext
import sys
import os

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DpoScriptArguments, init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, PeftConfig

from trl import (
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((DpoScriptArguments, TrainingArguments, ModelConfig))
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
    
    peft_config = PeftConfig.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True, attn_implementation="flash_attention_2")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )

    model.resize_token_embeddings(len(tokenizer))
        
    model = PeftModel.from_pretrained(model, model_config.model_name_or_path)
    model = model.merge_and_unload(safe_merge=True)

    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
 
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    def process(examples):
        
        INSTRUCTION = "Rewrite the given text and correct grammar, spelling, and punctuation errors."
        rejected_text = []
        chosen_text = []
        prompts = []
        print(f'len(examples) = {len(examples["prompt"])}')
        for i in range(len(examples["prompt"])):
            
            original_text = examples["prompt"][i]
            prompt = (
                f'### Instruction:\n'
                f'{INSTRUCTION}\n'
                f'### Input:\n{original_text}{tokenizer.eos_token}\n'
            )
            prompts.append(prompt)
            rejected =  (
                '### Response:\n'
                f'{examples["rejected"][i]}{tokenizer.eos_token}'
            )
            rejected_text.append(rejected)
            chosen = (
                '### Response:\n'
                f'{examples["chosen"][i]}{tokenizer.eos_token}'
            )
            chosen_text.append(chosen)
        examples["prompt"] = prompts
        examples["chosen"] = chosen_text
        examples["rejected"] = rejected_text
        return examples

    ds = ds.map(
        process,
        batched=True,
        load_from_cache_file=False,

    )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    ###############
    # Training
    ###############
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_ENTITY"] = "ay2324s2-cs4248-team-47"
    os.environ["WANDB_PROJECT"] = "dpo"
    with init_context:
        trainer = DPOTrainer(
            model,
            loss_type="hinge", # Remove if this is not rejection sampling dataset
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            max_prompt_length=args.max_prompt_length,
            generate_during_eval=args.generate_during_eval,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)