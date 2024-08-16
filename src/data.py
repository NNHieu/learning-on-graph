# import torch
import random
import copy
from datasets import load_dataset, load_from_disk
from .config import DataTrainingArguments, ModelArguments

def load_raw_dataset(data_args: DataTrainingArguments, model_args: ModelArguments):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_args.task_name = "star_graph"
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.load_from_disk:
            raw_datasets = load_from_disk(data_args.dataset_name)
            data_args.task_name = data_args.dataset_name.split('/')[-1]
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.
    return raw_datasets

def build_lego_tokenize_fn(data_args: DataTrainingArguments, tokenizer, text_column_name, max_target_length, val=False):
    def tokenize_fn(examples):
        input_ids = []
        labels = []
        attention_mask = []
        for line in examples[text_column_name]:
            prefix, target = line.strip().split('?')
            target = [prefix.split('/')[1][2]] + [p[-2] for p in target.split(':')]
            # target = [target[-2]]
            prefix += '?'

            prefix = tokenizer.encode(prefix)
            target = tokenizer.encode(target)

            if val:
                input_ids.append(prefix)
                attention_mask.append([1] * len(prefix))
                labels.append(target)
            else:
                if data_args.noisy_target > 0:
                    if random.random() < data_args.noisy_target:
                        nodes = filter(lambda x: isinstance(x, int) and x < 26, prefix)
                        nodes = list(set(nodes))
                        noisy_target = copy.deepcopy(target)
                        for i in range(1, len(noisy_target) - 1):
                            noisy_target[i] = random.choice(nodes)
                        seq = prefix + noisy_target
                    else:
                        seq = prefix + target
                else:
                    seq = prefix + target
                label = [-1] * len(prefix) + target            
                input_ids.append(seq)
                labels.append(label)
                attention_mask.append([1] * len(seq))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return tokenize_fn
            
def build_tokenize_function(data_args: DataTrainingArguments, tokenizer, text_column_name, max_target_length, val=False):
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    is_teacherless = data_args.objective == "teacherless"
    data_args.teacherless_token = tokenizer.encode('$')[0] if is_teacherless else None
    def tokenize_function(examples):
        input_ids = []
        labels = []
        attention_mask = []
        for line in examples[text_column_name]:
            prefix, target = line.strip().split('=')
            prefix += '='
            if data_args.objective == "reverse":
                target = ','.join(target.split(',')[::-1])
            prefix = tokenizer.encode(prefix)
            target = tokenizer.encode(target)
            # target = target[-4:-3]
            
            if val:
                if data_args.objective == "mixtask":
                    if random.random() >= data_args.mixtask_ratio:
                        pos = random.choice(list(range(2, len(target) - 1)))
                        target = [target[pos]]
                        # max_target_length = 1
                        prefix =  prefix[:-1] + tokenizer.encode("=" + "$"*pos)
                # if len(target) < max_target_length:
                #     target = target + [tokenizer.eos_token_id] * (max_target_length - len(target))
                
                input_ids.append(prefix)
                attention_mask.append([1] * len(prefix))
                labels.append(target)
                
            else:
                if is_teacherless and data_args.teacherless_token is not None:
                    teacherless_target = [data_args.teacherless_token] * len(target)
                    teacherless_target[0] = target[0]
                    seq = prefix + teacherless_target
                    label = [-1] * len(prefix) + target
                elif data_args.objective == "noisyguide" and data_args.noisy_target > 0:
                    if random.random() < data_args.noisy_target:
                        nodes = filter(lambda x: isinstance(x, int) and x < data_args.num_nodes, prefix)
                        nodes = list(set(nodes))
                        if not data_args.alter_label:
                            noisy_target = copy.deepcopy(target)
                            for i in range(1, len(noisy_target) - 1):
                                noisy_target[i] = random.choice(nodes)
                            seq = prefix + noisy_target
                    else:
                        seq = prefix + target
                    label = [-1] * len(prefix) + target
                elif data_args.objective == "noisytarget" and data_args.noisy_target > 0:
                    if random.random() < data_args.noisy_target:
                        nodes = filter(lambda x: isinstance(x, int) and x < data_args.num_nodes, prefix)
                        nodes = list(set(nodes))
                        pos = random.choice(list(range(1, len(target) - 2)))
                        target[pos] = random.choice(nodes)
                    seq = prefix + target
                    label = [-1] * len(prefix) + target
                elif data_args.objective == "mixtask":
                    r = random.random()
                    if r < data_args.mixtask_ratio:
                        seq = prefix + target
                    else:
                        pos = random.choice(list(range(2, len(target) - 1)))
                        target = [target[pos]]
                        prefix =  prefix[:-1] + tokenizer.encode("=" + "$"*pos)
                        seq = prefix + target
                    label = [-1] * len(prefix) + target
                else:
                    seq = prefix + target
                    label = [-1] * len(prefix) + target
                # label = torch.concatenate([-torch.ones_like(prefix), target], dim=-1).long()
                

                input_ids.append(seq)
                labels.append(label)
                attention_mask.append([1] * len(seq))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return tokenize_function