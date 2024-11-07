#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
from torch.nn import functional as F
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# from src.tokenizing import NumeralTokenizer, Tokenizer
from src.tokenizer import GPT2NumeralTokenizer
from src.config import ModelArguments, DataTrainingArguments
from src.data import load_raw_dataset, build_tokenize_function
from src.trainer.graph_trainer import CustomTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.44.0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# os.environ["WANDB_PROJECT"] = "star-graph"

def get_model_config(model_args, **kwargs):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        # "vocab_size": model_args.vocab_size,
        # "bos_token_id": model_args.vocab_size - 1,
        # "eos_token_id": model_args.vocab_size - 1,
        # "pad_token_id": model_args.vocab_size - 1,
    }
    config_kwargs.update(kwargs)
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    return config

def get_tokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        if model_args.tokenizer_name == "gpt":
            # t = NumeralTokenizer(model_args.num_nodes)
            # tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=model_args.num_nodes + 4, name='numeral')
            tokenizer = GPT2NumeralTokenizer(
                model_args.num_nodes,
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

def get_model(model_args, config, tokenizer):
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForCausalLM.from_config(config, 
                                                 trust_remote_code=model_args.trust_remote_code,
                                                #  attn_implementation="flash_attention_2", # https://github.com/huggingface/transformers/issues/30019
                                                )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"The embedding size is {embedding_size}")
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    return model

def check_checkpoint(model_args, training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.do_train:
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if checkpoint is not None:
            try:
                with open(os.path.join(checkpoint, "wandb_state.json"), "r", encoding="utf-8") as f:
                    text = f.read()
                wandb_state = json.loads(text)
                os.environ["WANDB_RUN_ID"] = wandb_state["run_id"]
                os.environ["WANDB_RESUME"] = "must"
                logger.info("Set WANDB_RUN_ID to %s" % os.environ["WANDB_RUN_ID"])
            except:
                pass
    return checkpoint

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args.num_nodes = data_args.num_nodes
    training_args.teacherless = data_args.teacherless

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_raw_dataset(data_args, model_args)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = get_tokenizer(model_args)
    config = get_model_config(model_args, 
                            #   vocab_size = len(tokenizer),
                            # bos_token_id = tokenizer.bos_token_id,
                            # eos_token_id = tokenizer.eos_token_id,
                            # pad_token_id = tokenizer.pad_token_id,
                        )
    model = get_model(model_args, config, tokenizer)

    training_args.run_name = ""
    training_args.run_name += f"{data_args.objective}"
    # if data_args.reverse:
    #     training_args.run_name += "reverse"
    # elif data_args.teacherless:
    #     training_args.run_name += "teacherless"
    # elif data_args.noisy_target:
    #     training_args.run_name += "noisy"
    #     if data_args.alter_label:
    #         training_args.run_name += "alter"
    # elif data_args.mixtask:
    #     training_args.run_name += "mixtask"
    # else:
    #     training_args.run_name += "teacherforcing"
    training_args.run_name += f"_{config.model_type}_{config.n_layer}x{config.n_head}x{config.n_embd}"
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name, training_args.run_name)
    logger.info(f"Output directory: {training_args.output_dir}")
    os.environ["WANDB_PROJECT"] = data_args.task_name

    checkpoint = check_checkpoint(model_args, training_args)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            # tokenized_datasets = raw_datasets.map(
            #     tokenize_function,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     remove_columns=column_names,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on dataset",
            # )
            max_target_length = max(raw_datasets["train"]["len_target"])
            max_target_length = max(max_target_length, max(raw_datasets["validation"]["len_target"]))

            raw_datasets["train"] = raw_datasets["train"].map(
                build_tokenize_function(data_args, tokenizer, text_column_name, max_target_length=max_target_length),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            raw_datasets["validation"] = raw_datasets["validation"].map(
                build_tokenize_function(data_args, tokenizer, text_column_name, max_target_length=max_target_length,val=True),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            raw_datasets["validation"] = raw_datasets["validation"].rename_column("labels", "targets")
            tokenized_datasets = raw_datasets
        else:
            tokenize_function = build_tokenize_function(data_args, tokenizer, text_column_name)
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    lm_datasets = tokenized_datasets
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            logger.info(f"Decode sample {index} of the training set: {tokenizer.decode(train_dataset[index]['input_ids'])}.")
            # Test forward pass
            # input_sample = train_dataset[0]
            # input_sample = {k:torch.tensor(v).unsqueeze(0) for k,v in input_sample.items()}
            # model(**input_sample)

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        logger.info(f"Features of the validation set: {eval_dataset.features}")
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
            logger.info(f"Decode sample {index} of the training set: {tokenizer.decode(eval_dataset[index]['input_ids'])}.")

        def preprocess_logits_for_metrics(logits, labels):
            # if isinstance(logits, tuple):
            #     # Depending on the model and config, logits may contain extra tensors,
            #     # like past_key_values, but logits always come first
            #     logits = logits[0]
            # print(logits)
            return logits.argmax(dim=-1)

        # metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels, inputs = eval_preds
            mask = (labels != tokenizer.eos_token_id)
            correct = (labels == preds) * mask
            completely_correct = (correct.sum(axis=1) == mask.sum(axis=1)).mean()
            token_acc = correct.sum(axis=0) / (mask.sum(axis=0) + 1e-9)

            metrics = {}
            for i in range(len(token_acc)):
                metrics[f"token_acc_{i}"] = token_acc[i]
            metrics["seq_acc"] = completely_correct
            return metrics

            # return {
            #     "seq_acc": completely_correct,
            #     # "first_token_acc": correct[:, 0].mean(),
            #     # "second_token_acc": correct[:, 1].mean(),
            #     # "target_token_acc": correct[:, -1].mean(),
            #     # "edge_acc": edge_acc.get(),
            #     # "final_edge_acc": final_edge_acc.get(),
            # }

    # Initialize our Trainer
    training_args.include_inputs_for_metrics = True
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        # data_collator=DataCollatorWithPadding(tokenizer, ),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )

    # Training
    if training_args.do_train:
        # checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()