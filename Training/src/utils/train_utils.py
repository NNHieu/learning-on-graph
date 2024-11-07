import os
import json
import random
from transformers.trainer_utils import get_last_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)


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

def build_metrics(tokenizer):
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

        print_idx = random.sample(range(len(labels)), 3)
        for i in print_idx:
            # input_decoded = tokenizer.decode(inputs[i])
            label_decoded = tokenizer.decode(labels[i])
            preds_decoded = tokenizer.decode(preds[i])
            # print("Input:", input_decoded)
            print("Prediction:", preds_decoded)
            print("Label:", label_decoded)

        return metrics
    return compute_metrics