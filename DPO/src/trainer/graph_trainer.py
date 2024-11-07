import json
import os
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Literal, Optional, Union, Tuple, List, Callable
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, IterableDataset, DataLoader

from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel, PreTrainedTokenizerBase
)
from trl import DPOTrainer
from trl.trainer.utils import pad_to_length
from transformers.utils import is_datasets_available, logging
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from transformers.integrations.tpu import tpu_spmd_dataloader

import wandb

if is_datasets_available():
    import datasets

logger = logging.get_logger(__name__)

def accuracy(logits, targets):
    # num_prefix_tokens = targets[0].eq(-1).sum().item()
    # num_target_tokens = targets.shape[1] - num_prefix_tokens
    # targets = targets[:, num_prefix_tokens:]
    # logits = logits[:, num_prefix_tokens:, :]
    correct = torch.argmax(logits, dim=-1).eq(targets).to(torch.float)
    seq_correct = (torch.sum(correct, dim=1) == (targets != -1).sum(dim=1)).float()
    acc = torch.mean(seq_correct)
    per_token_acc = correct.mean(dim=0)

    return acc, per_token_acc

class CustomTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None
        # logger.info("Input ids: %s", inputs["input_ids"])
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        lm_logits = outputs.logits
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
        
        # acc, token_acc = accuracy(shift_logits.detach(), shift_labels)
        # metrics = {"seq_acc": acc.item()}
        # for i in range(len(token_acc)):
        #     metrics[f"token_acc_{i}"] = token_acc[i].item()

        # # force log the metrics
        # self.store_metrics(metrics, train_eval="train")

        return (loss, outputs) if return_outputs else loss
    
    
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
    
    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor], max_new_tokens, do_sample=False):
        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        # generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        policy_output = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            output_logits=True,
            return_dict_in_generate=True,
        )
        # print(policy_output)
        # print(policy_output.logit)
        return policy_output
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        # prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        labels = inputs.pop("targets")
        # print(labels)
        num_target_tokens = labels.shape[1] 
        # num_prefix_tokens = labels.shape[1] - num_target_tokens
        # batch = {k: v[:, :num_prefix_tokens] for k, v in batch.items()}
        if self.args.teacherless:
            raise NotImplemented
            # pref_inputs = {}
            # # pref_inputs["input_ids"] = inputs["input_ids"][:, :num_prefix_tokens]
            # # pref_inputs["attention_mask"] = inputs["attention_mask"][:, :num_prefix_tokens]
            # outputs = model(**pref_inputs)
            # # first_token_logit = outputs.logits[:, -1]
            # generated_first_token = outputs.logits.argmax(dim=-1)[:, -1]
            # inputs['input_ids'][:, num_prefix_tokens] = generated_first_token
            # outputs = model(**inputs)
            # logits = outputs.logits[:, -num_target_tokens-1:-1]
        else:
            # inputs["input_ids"] = inputs["input_ids"][:, :num_prefix_tokens]
            # inputs["attention_mask"] = inputs["attention_mask"][:, :num_prefix_tokens]
            with torch.no_grad():
                # loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
                policy_output = self.get_batch_samples(model, inputs, max_new_tokens=num_target_tokens)
                # generated_ids, logits = policy_output
                logits = policy_output["logits"]
                logits = torch.stack(logits, dim=1)
            # preds = logits.argmax(dim=-1).cpu().numpy()
            # new_labels = np.ones(logits.shape[:2]) * -1
            # labels = labels.cpu().numpy()
            # for i in range(labels.shape[0]):
            #     l = (labels[i, :] != -1).sum()
            #     new_labels[i, :l] = labels[i,-l:]
            # correct = (new_labels == preds)
            # completely_correct = (correct.sum(axis=1) == (labels != -1).sum(axis=1)).mean()
            # token_acc = correct.sum(axis=0) / (labels != -1).sum(axis=0)
        return (torch.tensor(0, device=self.accelerator.device), logits, labels)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        result = super().save_model(output_dir, _internal_call)
        if output_dir is not None and wandb.run is not None:
                json_string = json.dumps({
                    "run_id": wandb.run.id
                }, indent=2, sort_keys=True) + "\n"
                with open(os.path.join(output_dir,"wandb_state.json"), "w", encoding="utf-8") as f:
                    f.write(json_string)
        return result

# class CustomDPOTrainer(DPOTrainer):
#     def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
#         """Generate samples from the model and reference model for the given batch of inputs."""

#         # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
#         # the torch cuda amp context manager as some hidden states are silently casted to full precision.
#         generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

#         with generate_context_manager():
#             policy_output = model.generate(
#                 input_ids=batch["prompt_input_ids"],
#                 attention_mask=batch["prompt_attention_mask"],
#                 max_length=self.max_length,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.pad_token_id,
#             )

#             # if reference_output in batch use that otherwise use the reference model
#             if "reference_output" in batch:
#                 reference_output = batch["reference_output"]
#             else:
#                 if self.ref_model is None:
#                     with self.null_ref_context():
#                         reference_output = self.model.generate(
#                             input_ids=batch["prompt_input_ids"],
#                             attention_mask=batch["prompt_attention_mask"],
#                             max_length=self.max_length,
#                             do_sample=True,
#                             pad_token_id=self.tokenizer.pad_token_id,
#                         )
#                 else:
#                     reference_output = self.ref_model.generate(
#                         input_ids=batch["prompt_input_ids"],
#                         attention_mask=batch["prompt_attention_mask"],
#                         max_length=self.max_length,
#                         do_sample=True,
#                         pad_token_id=self.tokenizer.pad_token_id,
#                     )

#         policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
#         policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

#         reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
#         reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

#         return policy_output_decoded, reference_output_decoded
    
#     def prediction_step(
#         self,
#         model: Union[PreTrainedModel, nn.Module],
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None,
#     ):
#         if not self.use_dpo_data_collator:
#             warnings.warn(
#                 "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
#                 "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
#             )
#         if ignore_keys is None:
#             if hasattr(model, "config"):
#                 ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
#             else:
#                 ignore_keys = []

#         prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

#         with torch.no_grad(), prediction_context_manager():
#             loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

#         # force log the metrics
#         self.store_metrics(metrics, train_eval="eval")

#         if prediction_loss_only:
#             return (loss.detach(), None, None)

#         # logits for the chosen and rejected samples from model
#         logits_dict = {
#             "eval_logits/chosen": metrics["eval_logits/chosen"],
#             "eval_logits/rejected": metrics["eval_logits/rejected"],
#         }
#         logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
#         logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
#         labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

#         return (loss.detach(), logits, labels)