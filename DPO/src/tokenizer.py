"""Based on tokenization classes for OpenAI GPT."""

import json
import os
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Tuple, Union


import regex as re

from transformers.tokenization_utils  import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

from transformers.tokenization_utils_base import (
    PaddingStrategy, 
    EncodedInput, 
    TensorType, 
    BatchEncoding,
    is_tf_tensor,
    is_torch_tensor,
    to_py_obj,
    Sized
)

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class NumeralTokenizer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        # Define encoder and decoder as a dictionary
        self.encoder = {str(i): i for i in range(num_nodes)}
        self.encoder['|'] = num_nodes
        self.encoder['='] = num_nodes + 1
        self.encoder['/'] = num_nodes + 2
        self.encoder['$'] = num_nodes + 3
        self.encoder[':'] = -1

        self.decoder = {i: str(i) for i in range(num_nodes)}
        self.decoder[num_nodes] = '|'
        self.decoder[num_nodes + 1] = '='
        self.decoder[num_nodes + 2] = '/'
        self.decoder[num_nodes + 3] = '$'
        self.decoder[-1] = ':'

    def encode(self, x):
        out = []
        i = 0
        while i < len(x):
            if x[i] == ',':
                i += 1
                continue
            s = ''
            j = 0
            while i + j < len(x) and x[i + j] in numbers:
                s += x[i + j]
                j += 1
            if s == '':
                s = x[i]
                i += 1
            else:
                i += j
            # out.append(self.encoder[s])
            out.append(s)


        return out

    def decode(self, x):
        return [self.decoder[i] for i in x]
    
    def is_node(self, t):
        return isinstance(t, int) and t < self.num_nodes


class GPT2NumeralTokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GPT2Tokenizer

    >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        num_nodes,
        # merges_file,
        errors="replace",
        unk_token="#",
        bos_token="#",
        eos_token="#",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token

        # with open(vocab_file, encoding="utf-8") as vocab_handle:
        #     self.encoder = json.load(vocab_handle)
        # self.decoder = {v: k for k, v in self.encoder.items()}

        num_tokenizer = NumeralTokenizer(num_nodes)
        self.encoder = num_tokenizer.encoder
        self.decoder = num_tokenizer.decoder
        self.num_tokenizer = num_tokenizer


        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # with open(merges_file, encoding="utf-8") as merges_handle:
        #     bpe_merges = merges_handle.read().split("\n")[1:-1]
        # bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        bpe_merges = []
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        """Tokenize a string."""
        # bpe_tokens = []
        # for token in re.findall(self.pat, text):
        #     token = "".join(
        #         self.byte_encoder[b] for b in token.encode("utf-8")
        #     )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
        #     bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # return bpe_tokens
        tokens = self.num_tokenizer.encode(text)
        # tokens = [str(self.decoder[t]) for t in tokens]
        # print(tokens)
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = ",".join(tokens)
        text = text.replace(",|,", "|")
        text = text.replace(",/,", "/")
        text = text.replace(",=", "=")
        text = text.replace("=,", "=")
        text = text.replace(",:", ":")
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)
    
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults

        return_labels = "labels" in encoded_inputs
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if return_labels:
                    encoded_inputs["labels"] = encoded_inputs["labels"] + [-1] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if return_labels:
                    encoded_inputs["labels"] = [-1] * difference + encoded_inputs["labels"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError(f"Invalid padding strategy:{self.padding_side}")

        return encoded_inputs

if __name__ == "__main__":
    tokenizer = GPT2NumeralTokenizer(50)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(tokenizer.vocab_size)
    print(len(tokenizer))
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.added_tokens_encoder)

    enc = tokenizer(["19,10|15,8|5,15|15,6|18,12|18,0|15,7|12,6|2,0|2,4|7,17|18,9|10,13|16,17|12,9|6,18|15,5|5,7|12,11|19,16|12,8|12,5|0,11|0,7|2,10|13,17|0,17|16,19|15,11|12,15|18,1|3,1|17,7|17,2|10,16|14,3|18,15|19,3|2,16|1,11|1,9|2,17|0,13|10,19|7,18|0,2|5,9|15,12|13,0|4,19|16,10|15,18|13,16|16,2|10,4|5,8|1,5|9,11|18,7|6,5|0,10|8,14|16,13|5,11|17,13|1,6|11,9|10,17|10,0|10,3|14,1|13,2|3,14|18,5|0,19|12,18|14,8|1,8|2,19|16,0|13,10|0,16|15,9|4,10|2,13|5,1|9,18|1,14|1,12|15,1|8,12|6,12|16,9|1,3|0,9|10,9|3,8|8,3|5,18|12,1|12,3|5,6|3,9|1,18|10,2|2,7|16,4|5,12|18,6|8,1|11,12|18,11|19,14|1,19|6,15|14,19/12,16=12,1,19,16",
                            "17,16|11,18|18,6|12,15|14,16|19,16|14,2|11,7|4,0|5,1|0,2|11,6|10,12|10,17|5,10|3,10|19,17|7,15|15,16|13,10|7,14|11,16|0,11|13,9|0,14|11,8|12,7|17,7|3,9|12,17|3,13|14,11|14,19|3,1|3,15|16,15|17,10|17,5|11,19|0,13|14,8|5,16|0,6|0,4|6,0|17,15|10,3|4,6|4,18|7,2|2,12|10,15|16,10|2,14|18,13|0,18|10,16|12,16|11,14|7,11|14,18|13,16|17,12|18,4|3,16|5,8|2,8|5,17|19,14|14,7|5,2|17,1|14,12|12,5|1,15|9,13|11,4|7,19|7,12|19,11|2,7|11,2|19,7|6,4|10,1|19,8|5,12|1,17|6,18|7,8|5,7|7,16|13,3|19,12|8,7|18,0|8,2|4,13|17,3|7,10|19,2|9,3|17,2|19,5|2,11|18,11|11,12|5,15/4,15=4,13,16,15"])
    print(enc)
    # dec = tokenizer.decode(enc)
    # print(dec)
    paded_enc = tokenizer.pad(enc)
    print(paded_enc)

    dec = tokenizer.decode(paded_enc["input_ids"][1])
    print(dec)
    print(len(paded_enc["input_ids"][1]))