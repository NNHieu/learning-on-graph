# import torch
import random
import copy
from datasets import load_dataset, load_from_disk
from .config import DataTrainingArguments, ModelArguments, LegoDataTrainingArguments, TrainingArguments
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils.logging import get_logger
import itertools

logger = get_logger(__name__)

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

def build_lego_tokenize_fn(data_args: LegoDataTrainingArguments, tokenizer, text_column_name, val=False):
    def tokenize_fn(examples):
        input_ids = []
        labels = []
        attention_mask = []
        for eidx, line in enumerate(examples[text_column_name]):
            prefix, target = line.strip().split('?')
            if data_args.task == "steps":
                target = examples['target_steps'][eidx]
                prefix += '?'
            elif data_args.task == "direct":
                target = examples['target_direct'][eidx]
                prefix += '?'
            elif data_args.task == "path":
                target = examples['target_path'][eidx]
                prefix += '?'
            elif data_args.task == "verify_steps":
                target = examples['target_verify'][eidx]
                sol, is_correct = target.split('?')
                prefix += '?' + sol + '?'
                target = is_correct
            prefix = tokenizer.encode(prefix)
            target = tokenizer.encode(target)

            if val:
                input_ids.append(prefix)
                attention_mask.append([1] * len(prefix))
                labels.append(target)
            else:
                if data_args.objective == "noisytarget":
                    assert data_args.task != "direct", "Noisy target is not supported for direct task"
                    assert data_args.task != "verify_steps", "Noisy target is not supported for verify_steps task"
                    if random.random() < data_args.noisy_target:
                        if data_args.task == "path":
                            nodes = filter(lambda x: isinstance(x, int) and x < 26, prefix)
                            nodes = list(set(nodes))
                        else:
                            nodes = tokenizer.encode("-+")
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
        for idx, line in enumerate(examples[text_column_name]):
            prefix, target = line.strip().split('=')
            prefix += '='
            if data_args.objective == "reverse":
                target = ','.join(target.split(',')[::-1])
            prefix = tokenizer.encode(prefix)
            target = tokenizer.encode(target)
            # target = target[-4:-3]
            
            if val:
                if data_args.objective == "mixtask":
                    if random.random() < data_args.mixtask_ratio:
                        # pos = random.choice(list(range(2, len(target) - 1)))
                        pos = len(target) - 2
                        target = [target[pos]]
                        prefix =  prefix[:-1] + tokenizer.encode("&")
                elif data_args.objective == "improving":
                    nodes = filter(lambda x: isinstance(x, int) and x < data_args.num_nodes, prefix)
                    nodes = list(set(nodes))
                    pause = tokenizer.encode('$')[0]
                    nodes.append(pause)
                    k = random.randint(0, len(target))
                    noisy_target = copy.deepcopy(target)
                    improving_target = tokenizer.encode('.' * len(target))
                    if k > 0:
                        pos = list(range(len(target)))
                        random.shuffle(pos)
                        pos = pos[:k]
                        for p in pos:
                            noisy_target[p] = random.choice(nodes)
                            improving_target[p] = target[p]
                    prefix = prefix + noisy_target + tokenizer.encode(">")
                    target = improving_target
                elif data_args.objective == "error_detection":
                    nodes = filter(lambda x: isinstance(x, int) and x < data_args.num_nodes, prefix)
                    nodes = list(set(nodes))
                    error_rate = 0.4
                    target_len = len(target)
                    num_error = int(error_rate * target_len)
                    assert num_error > 0
                    alter_idxs = random.sample(range(target_len), num_error)
                    
                    noisy_target = copy.deepcopy(target)
                    detection_target = [tokenizer.encode("+")[0]] * target_len
                    for i in alter_idxs:
                        detection_target[i] = tokenizer.encode("-")[0]
                        alter_node = target[i]
                        while alter_node == target[i]:
                            alter_node = random.choice(nodes)
                        noisy_target[i] = alter_node
                    detection_target = list(itertools.chain.from_iterable(zip(noisy_target, detection_target))) + tokenizer.encode(">") + target
                    prefix = prefix + noisy_target + tokenizer.encode(">")
                    target = detection_target
                elif data_args.objective == "step_eval":
                    # wrong_path = tokenizer.encode(examples['wrong_path'][idx])
                    wrong_path = [tokenizer.encode(p) for p in examples['wrong_path'][idx]][0]
                    # node_labels = list(zip(wrong_path, tokenizer.encode('-' * len(wrong_path)))) + list(zip(target, tokenizer.encode('+' * len(wrong_path))))
                    # exposure_rate = 0.5
                    # num_demonstrations = int(len(node_labels) * exposure_rate)
                    # inds = random.sample(range(len(node_labels)), num_demonstrations)
                    
                    step_eval_target = wrong_path + tokenizer.encode('-') + target
                    prefix = prefix[:-1] + tokenizer.encode("<") + step_eval_target[:2]
                    target =  step_eval_target[2:] + tokenizer.encode(">") + target 
                if len(target) < max_target_length:
                    target = target + [tokenizer.eos_token_id] * (max_target_length - len(target))
                
                input_ids.append(prefix)
                attention_mask.append([1] * len(prefix))
                labels.append(target + [tokenizer.eos_token_id])
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
                    if r > data_args.mixtask_ratio:
                        seq = prefix + target
                    else:
                        # pos = random.choice(list(range(2, len(target) - 1)))
                        pos = len(target) - 2
                        target = [target[pos]]
                        prefix =  prefix[:-1] + tokenizer.encode("&")
                        seq = prefix + target
                    label = [-1] * len(prefix) + target
                elif data_args.objective == "fit_prefix":
                    seq = prefix + target
                    label = seq
                elif data_args.objective == "improving":
                    nodes = filter(lambda x: isinstance(x, int) and x < data_args.num_nodes, prefix)
                    nodes = list(set(nodes))
                    pause = tokenizer.encode('$')[0]
                    nodes.append(pause)
                    k = random.randint(0, len(target))
                    noisy_target = copy.deepcopy(target)
                    improving_target = tokenizer.encode('.' * len(target))
                    if k > 0:
                        pos = list(range(len(target)))
                        random.shuffle(pos)
                        pos = pos[:k]
                        for p in pos:
                            noisy_target[p] = random.choice(nodes)
                            improving_target[p] = target[p] if random.random() < 0.5 else pause
                    seq = prefix + noisy_target + tokenizer.encode(">")
                    label = [-1] * len(seq) + improving_target
                    seq = seq + improving_target
                elif data_args.objective == "error_detection":
                    # nodes = filter(lambda x: isinstance(x, int) and x < data_args.num_nodes, prefix)
                    # nodes = list(set(nodes))
                    # error_rate = 0.4
                    wrong_path = tokenizer.encode(examples['wrong_path'][idx])

                    target_len = len(target)
                    noisy_target = copy.deepcopy(target)
                    detection_target = "+"* target_len

                    num_error = random.choice(range(target_len-1))
                    # assert num_error > 0
                    if num_error > 0:
                        alter_idxs = random.sample(range(1,target_len), num_error)
                        for i in alter_idxs:
                            detection_target[i] ="-"
                            # alter_node = target[i]
                            # while alter_node == target[i]:
                            #     alter_node = random.choice(nodes)
                            alter_node = wrong_path[i]
                            noisy_target[i] = alter_node
                    detection_target = tokenizer.encode(detection_target)
                    detection_target = list(itertools.chain.from_iterable(zip(noisy_target, detection_target))) + tokenizer.encode(">") + target
                    prefix = prefix + noisy_target + tokenizer.encode(">") 
                    seq = prefix + detection_target 
                    label = [-1] * len(prefix) + detection_target
                elif data_args.objective == "step_eval":
                    wrong_paths = [tokenizer.encode(p) for p in examples['wrong_path'][idx]]
                    # node_labels = list(zip(wrong_path, tokenizer.encode('-' * len(wrong_path)))) + list(zip(target, tokenizer.encode('+' * len(wrong_path))))
                    # exposure_rate = random.random() * 0.7 + 0.2
                    # num_demonstrations = int(len(node_labels) * exposure_rate)
                    # inds = random.sample(range(len(node_labels)), num_demonstrations)

                    num_exp = random.choice(range(len(wrong_paths) + 1))
                    random.shuffle(wrong_paths)
                    step_eval_target = []
                    for i in range(num_exp):
                        step_eval_target += wrong_paths[i] + tokenizer.encode('-')
                    step_eval_target += target



                    # if random.random() < 0.5:
                    #     step_eval_target = target
                    # else:
                    #     # cap_idx = random.choice(range(2,len(wrong_path) + 1))
                    #     cap_idx = len(wrong_path)
                    #     step_eval_target = wrong_path[:cap_idx] + tokenizer.encode('-') + target

                    prefix = prefix[:-1]
                    target = tokenizer.encode('<') + step_eval_target + tokenizer.encode('>') + target
                    seq = prefix + target
                    label = [-1] * len(prefix) + target
                elif data_args.objective == "step_eval_short":
                    wrong_paths = [tokenizer.encode(p) for p in examples['wrong_path'][idx]]
                    num_exp = random.choice(range(len(wrong_paths) + 1))
                    random.shuffle(wrong_paths)
                    step_eval_target = []
                    for i in range(num_exp):
                        step_eval_target += wrong_paths[i] + tokenizer.encode('-')
                    step_eval_target += target
                    prefix = prefix[:-1]
                    target = tokenizer.encode('<') + step_eval_target + tokenizer.encode('>') + target
                    seq = prefix + target
                    label = [-1] * len(prefix) + target
                else:
                    seq = prefix + target
                    label = [-1] * len(prefix) + target
                # label = torch.concatenate([-torch.ones_like(prefix), target], dim=-1).long()
                

                input_ids.append(seq + [tokenizer.eos_token_id])
                labels.append(label + [tokenizer.eos_token_id])
                attention_mask.append([1] * len(seq) + [1])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    return tokenize_function

class LEGODataModule:
    def __init__(self):
        self._train_dataset = None
        self._eval_dataset = None
        self._tokenized_datasets = None

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def tokenized_datasets(self):
        return self._tokenized_datasets

    def setup(self, data_args: LegoDataTrainingArguments, model_args: ModelArguments, training_args: TrainingArguments, tokenizer: AutoTokenizer):
        raw_datasets = load_raw_dataset(data_args, model_args)
        column_names = list(raw_datasets["train" if training_args.do_train else "validation"].features)
        text_column_name = "symbol_text"

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                def tokenize_dataset(dataset, is_validation=False):
                    tokenize_fn = build_lego_tokenize_fn(data_args, tokenizer, text_column_name, val=is_validation)
                    return dataset.map(
                        tokenize_fn,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc=f"Tokenizing {'validation' if is_validation else 'train'} dataset",
                    )

                raw_datasets["train"] = tokenize_dataset(raw_datasets["train"])
                raw_datasets["validation"] = tokenize_dataset(raw_datasets["validation"], is_validation=True)
                raw_datasets["validation"] = raw_datasets["validation"].rename_column("labels", "targets")
                
                self._tokenized_datasets = raw_datasets
            else:
                tokenize_fn = build_lego_tokenize_fn(data_args, tokenizer, text_column_name)
                self._tokenized_datasets = raw_datasets.map(
                    tokenize_fn,
                    batched=True,
                    remove_columns=column_names,
                )
        
        if training_args.do_train:
            if "train" not in self._tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = self._tokenized_datasets["train"]
            max_train_samples = len(train_dataset)
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
                logger.info(f"Decode sample {index} of the training set: {tokenizer.decode(train_dataset[index]['input_ids'])}.")

            self._train_dataset = train_dataset

        if training_args.do_eval:
            if "validation" not in self._tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = self._tokenized_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

            # Log a few random samples from the validation set:
            for index in random.sample(range(len(eval_dataset)), 3):
                logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
                logger.info(f"Decode sample {index} of the validation set: {tokenizer.decode(eval_dataset[index]['input_ids'])}.")

            self._eval_dataset = eval_dataset


class ShortestDataModule:
    def __init__(self):
        self._train_dataset = None
        self._eval_dataset = None
        self._tokenized_datasets = None

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def tokenized_datasets(self):
        return self._tokenized_datasets

    def setup(self, data_args: LegoDataTrainingArguments, model_args: ModelArguments, training_args: TrainingArguments, tokenizer: AutoTokenizer, config):
        raw_datasets = load_raw_dataset(data_args, model_args)
        logger.info(f"Raw train set size: {len(raw_datasets['train'])}")
        logger.info(f"Raw val set size: {len(raw_datasets['validation'])}")

        column_names = list(raw_datasets["train" if training_args.do_train else "validation"].features)

        max_target_length = 50
        def tokenize_fn(examples, is_validation: bool = False):
            input_ids = []
            attention_masks = []
            labels = []
            for i in range(len(examples['query'])):
                query = examples['query'][i] + ("<" if data_args.objective != "teacherforcing" else "=")
                reverse_target = "-" + ",".join(reversed(examples['target'][i].split(',')))
                if data_args.objective == "step_eval":
                    generation = examples['steps'][i] + reverse_target + ">=" + examples['target'][i]
                elif data_args.objective == "teacherforcing":
                    generation = examples['target'][i]
                elif data_args.objective == "pause":
                    generation = ','.join([s.split(",")[0] + "($)"for s in examples['trace'][i]]) + reverse_target + ">=" + examples['target'][i]

                query_ids = [tokenizer.bos_token_id] + tokenizer.encode(query)
                generation_ids = tokenizer.encode(generation) + [tokenizer.eos_token_id]

                if not is_validation:
                    if data_args.aug == "padding":
                        query_ids += [tokenizer.eos_token_id] * random.randint(0, 512)
                    elif data_args.aug == "none":
                        pass
                    else:
                        raise ValueError

                    ids = query_ids + generation_ids
                    input_ids.append(ids)
                    attention_masks.append([1] * len(ids))
                    labels.append([-1] * len(query_ids) + generation_ids)
                else:
                    generation_ids = generation_ids[:max_target_length]
                    if len(generation_ids) < max_target_length:
                        generation_ids += [tokenizer.eos_token_id] * (max_target_length - len(generation_ids))
                    input_ids.append(query_ids)
                    attention_masks.append([1] * len(query_ids))
                    labels.append(generation_ids)
                
            return {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
            }

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                def tokenize_dataset(dataset, is_validation=False):
                    return dataset.map(
                        lambda x: tokenize_fn(x, is_validation=is_validation),
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc=f"Tokenizing {'validation' if is_validation else 'train'} dataset",
                    )

                raw_datasets["train"] = tokenize_dataset(raw_datasets["train"])
                logger.info(f"Tokenized train dataset size: {len(raw_datasets['train'])}")
                raw_datasets["validation"] = tokenize_dataset(raw_datasets["validation"], is_validation=True)
                raw_datasets["validation"] = raw_datasets["validation"].rename_column("labels", "targets")
                
                self._tokenized_datasets = raw_datasets
            else:
                self._tokenized_datasets = raw_datasets.map(
                    tokenize_fn,
                    batched=True,
                    remove_columns=column_names,
                )

        if hasattr(config, "max_position_embeddings"):
            max_pos_embeddings = config.max_position_embeddings
        else:
            # Define a default value if the attribute is missing in the config.
            max_pos_embeddings = 1024
        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > max_pos_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                if max_pos_embeddings > 0:
                    block_size = min(1024, max_pos_embeddings)
                else:
                    block_size = 1024
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            # concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
            # total_length = len(concatenated_examples[list(examples.keys())[0]])
            # # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            # total_length = (total_length // block_size) * block_size
            # # Split by chunks of max_len.
            # result = {
            #     k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            #     for k, t in concatenated_examples.items()
            # }

            result = {
                k: [[]] for k in examples.keys()
            }
            first_k = list(examples.keys())[0]
            for i in range(len(examples[first_k])):
                if len(result[first_k][-1]) + len(examples[first_k][i]) <= block_size:
                    for k in examples.keys():
                        result[k][-1].extend(examples[k][i])
                else:
                    for k in examples.keys():
                        result[k].append(examples[k][i])
                


            # result["labels"] = result["input_ids"].copy()
            return result
        
        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                self._tokenized_datasets["train"] = self._tokenized_datasets["train"].map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                self._tokenized_datasets["train"] = self._tokenized_datasets["train"].map(
                    group_texts,
                    batched=True,
                )


        
        if training_args.do_train:
            if "train" not in self._tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = self._tokenized_datasets["train"]
            logger.info(f"Train set size: {len(train_dataset)}")
            max_train_samples = len(train_dataset)
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
                logger.info(f"Decode sample {index} of the training set: {tokenizer.decode(train_dataset[index]['input_ids'])}.")

            self._train_dataset = train_dataset

        if training_args.do_eval:
            if "validation" not in self._tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = self._tokenized_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

            # Log a few random samples from the validation set:
            logger.info(f"Validation set size: {len(eval_dataset)}")    
            for index in random.sample(range(len(eval_dataset)), 3):
                logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
                logger.info(f"Decode sample {index} of the validation set: {tokenizer.decode(eval_dataset[index]['input_ids'])}.")

            self._eval_dataset = eval_dataset
