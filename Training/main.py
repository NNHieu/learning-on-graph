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

import logging
import os
import sys
import warnings

import torch

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorWithPadding,
    is_torch_xla_available,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from src.config import ModelArguments, LegoDataTrainingArguments as DataTrainingArguments
from src.utils.logging import setup_logging
from src.tokenizers import GPT2LegoTokenizer, GPT2NumeralTokenizer
# We need to setup the logging before importing these modules
from src.data import LEGODataModule, ShortestDataModule
from src.graph_trainer import CustomTrainer
from src.utils.train_utils import check_checkpoint, build_metrics
from src.utils.logging import setup_transformers_logging, get_logger

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.40.0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = get_logger(__name__)


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

def get_tokenizer(model_args, dataset_name):
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
            if "lego" in dataset_name:
                tokenizer = GPT2LegoTokenizer(
                    padding_side='left'
                )
            elif "shortest" in dataset_name:
                tokenizer = GPT2NumeralTokenizer(
                    50,
                    padding_side='left'
                )
            else:
                raise ValueError
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
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # if model_args.resize_embedding:
    #     model.resize_token_embeddings(len(tokenizer))
    #     logger.info(f"Resized the embedding layer")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"The embedding size is {embedding_size}")
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    return model

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
    setup_transformers_logging(transformers_log_level=training_args.get_process_log_level(), should_log=training_args.should_log)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    is_gpt2 = ((model_args.model_name_or_path is not None) and ("gpt2" in model_args.model_name_or_path)) or ((model_args.config_name is not None) and ("gpt2" in model_args.config_name))
    tokenizer = get_tokenizer(model_args, data_args.dataset_name)
    config = get_model_config(model_args, 
                            attn_implementation="flash_attention_2" if not is_gpt2 else None,
                            bos_token_id = tokenizer.bos_token_id,
                            eos_token_id = tokenizer.eos_token_id,
                            vocab_size = tokenizer.vocab_size,
                        )
    model = get_model(model_args, config, tokenizer)

    task_name = data_args.dataset_name.split('/')[-1]
    training_args.run_name = f"{data_args.task}_"
    training_args.run_name += f"{data_args.objective}"
    if data_args.aug != "none":
        training_args.run_name += f"{data_args.aug}"
    try:
        training_args.run_name += f"_{config.model_type}_{config.n_layer}x{config.n_head}x{config.n_embd}"
    except AttributeError:
        training_args.run_name += f"_{config.model_type}_{config.num_hidden_layers}x{config.num_attention_heads}x{config.hidden_size}"
    # training_args.run_name += f"_{config.model_type}_{config.n_layer}x{config.n_head}x{config.n_embd}"
    training_args.output_dir = os.path.join(training_args.output_dir, task_name, training_args.run_name)
    logger.info(f"Output directory: {training_args.output_dir}")
    os.environ["WANDB_PROJECT"] = task_name

    checkpoint = check_checkpoint(model_args, training_args)

    # Initialize the LEGODataModule
    if "lego" in data_args.dataset_name:
        data_module = LEGODataModule()
    elif "shortest" in data_args.dataset_name:
        data_module = ShortestDataModule()
    
    # Setup the data module
    data_module.setup(data_args, model_args, training_args, tokenizer, config)
    
    # Get the datasets
    train_dataset = data_module.train_dataset if training_args.do_train else None
    eval_dataset = data_module.eval_dataset if training_args.do_eval else None
    
    # Initialize our Trainer
    training_args.include_inputs_for_metrics = True
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_metrics(tokenizer) if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1)
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        train_result.metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save model card
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