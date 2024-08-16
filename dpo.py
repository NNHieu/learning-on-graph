#!/usr/bin/env python
# coding=utf-8
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
import random
import sys
import os

import torch
import datasets
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    # ModelArguments,
    # apply_chat_template,
    # decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    # get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer

from src.tokenizer import GPT2NumeralTokenizer
from src.dpo_config import ModelArguments, DataTrainingArguments
from src.data import load_raw_dataset, build_tokenize_function
from src.trainer.graph_trainer import CustomTrainer

# from huggingface_hub import login as hf_login
from dotenv import dotenv_values
env_config = dotenv_values(".env")
# hf_login(token=env_config["HF_TOKEN"])

logger = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"] = "star-graph-dpo"
os.environ["WANDB_RESUME"] = "must"
os.environ["WANDB_RUN_ID"] = "1sehxtpq"

def get_tokenizer(model_args, data_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        if model_args.tokenizer_name == "gpt":
            tokenizer = GPT2NumeralTokenizer(
                data_args.num_nodes,
                padding_side='left'
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    # elif model_args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer

def main():
    parser = H4ArgumentParser((ModelArguments, DataTrainingArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    # raw_datasets = get_datasets(
    #     data_args,
    #     splits=data_args.dataset_splits,
    #     configs=data_args.dataset_configs,
    #     columns_to_keep=["text", "wrong_path", "chosen", "rejected", "prompt", "completion", "label"],
    # )
    # logger.info(
    #     f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    raw_datasets = datasets.load_dataset("nnheui/star-d2_p5_n50")
    max_target_length = max(raw_datasets["train"]["len_target"])
    max_target_length = max(max_target_length, max(raw_datasets["validation"]["len_target"]))
    # column_names = list(raw_datasets["train"].features)
    def preference_prepair(example):
        prefix, target = example['text'].strip().split('=')
        prefix += "="
        return {
            "prompt": prefix,
            "chosen": target,
            "rejected": example['wrong_path']
        }
    raw_datasets = raw_datasets.map(preference_prepair, 
                                    # remove_columns=column_names
                                    )
    # raw_datasets = raw_datasets.filter(lambda x: x["chosen"] != x["rejected"])

    #####################################
    # Load tokenizer and process datasets
    #####################################
    # data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    # quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # use_flash_attention_2=model_args.use_flash_attention_2,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            # use_flash_attention_2=model_args.use_flash_attention_2,
            # device_map=get_kbit_device_map() if quantization_config is not None else None,
            # quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        mask = (labels != tokenizer.eos_token_id)
        correct = (labels == preds) * mask
        completely_correct = (correct.sum(axis=1) == mask.sum(axis=1)).mean()
        token_acc = correct.sum(axis=0) / (mask.sum(axis=0) + 1e-9)

        metrics = {}
        for i in range(len(token_acc)):
            metrics[f"token_acc_{i}"] = token_acc[i]
        metrics["seq_acc"] = completely_correct
        return metrics

    # if model_args.use_peft is True:
    #     ref_model = None
    #     ref_model_kwargs = None
    
    #########################
    # Instantiate DPO trainer
    #########################
    max_length = tokenizer([raw_datasets["train"][0]["prompt"]], return_tensors="pt")["input_ids"].shape[1] + max_target_length
    print(max_length)
    training_args.generate_during_eval = False
    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=training_args.max_prompt_length,
        # peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        # Added
        # precompute_ref_log_probs=True,
        # compute_metrics=compute_metrics,

    )

    # ###############
    # # Training loop
    # ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # ##################################
    # # Save model and create model card
    # ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # ##########
    # # Evaluate
    # ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
