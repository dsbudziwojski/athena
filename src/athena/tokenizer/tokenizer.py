from tokenizers import Tokenizer, normalizers, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents
from athena.tokenizer.utilities import json_files_iter

class GeneralTokenizer(object):
    """
    A flexible tokenizer interface that supports both a HuggingFace Tokenizers library-based BPE
    implementation (when tokenizer_type='library') and a custom implementation stub (when
    tokenizer_type='default').

    Attributes:
        _text_type (str): Input data format ('list' or 'files').
        _tokenizer_type (str): Backend choice ('default' for custom, 'library' for BPE via HF).
        _model_type (str): Model architecture ('BPE').
        tokenizer (Tokenizer): HuggingFace Tokenizer instance when using library backend.
        _trainer (BpeTrainer): Trainer for list- or file-based BPE training (library only).
    """
    def __init__(self, text_type="list", tokenizer_type="library", model_type="BPE"):
        """
        Initialize the GeneralTokenizer.

        Args:
            text_type (str): Data source type ('list' for in-memory lists, 'files' for JSON files).
            tokenizer_type (str): Which backend to use ('library' for HF BPE or 'default' as a stub).
            model_type (str): Model architecture ('BPE').

        Raises:
            Exception: If tokenizer_type, model_type, or text_type is unsupported.
        """
        if tokenizer_type not in ["default", "library"]:
            raise Exception("Invalid tokenizer")
        if model_type not in ["BPE"]:
            raise Exception("Invalid architecture")

        self._text_type = text_type
        self._tokenizer_type = tokenizer_type
        self._model_type = model_type

        if tokenizer_type == "library":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.tokenizer.enable_truncation(max_length=128)
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=0,
                pad_type_id=0,
                pad_token="[PAD]",
                length=128,
                pad_to_multiple_of=64
            )
            if text_type == "list" or text_type == "files":
                self._trainer = trainers.BpeTrainer(
                    vocab_size=30_000,
                    min_frequency=2,
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                )
            else:
                raise Exception("Invalid text_type")

    def train_tokenizer(self, data, messages=False):
        """
        Train the tokenizer on provided data using the configured backend.

        Args:
            data (Iterable[str]): Text strings if text_type='list', or file paths if text_type='files'.

        Raises:
            Exception: If invalid text_type or non-library backend.
        """
        if messages:
            print("TRAINING TOKENIZER STARTING")
        if self._tokenizer_type == "library":
            if self._text_type == "list":
                self.tokenizer.train_from_iterator(data, self._trainer)
            elif self._text_type == "files":
                self.tokenizer.train_from_iterator(json_files_iter(data), self._trainer)
            else:
                raise Exception("Invalid text_type")
        else:
            raise Exception("Other Training Tokenizer Types Not Supported Yet")
        if messages:
            print("TRAINING TOKENIZER COMPLETE")

    def encode(self, data, data_type="batch"):
        if data_type == "batch":
            if self._tokenizer_type == "library":
                vectorized_data = self.tokenizer.encode_batch(data)
                return [enc.ids for enc in vectorized_data]
            else:
                raise Exception("Other Tokenizer Types Not Supported Yet")
        elif data_type == "list":
            if self._tokenizer_type == "library":
                vectorized_data = []
                for record in data:
                    output = self.tokenizer.encode(record)
                    print(output.tokens)
                    vectorized_data.append(list(output.ids))
                return vectorized_data
            else:
                raise Exception("Other Tokenizer Types Not Supported Yet")
        else:
            raise Exception("Other data_type Not Supported Yet")

    def get_vocab_size(self):
        if self._tokenizer_type == "library":
            return self.tokenizer.get_vocab_size()
        else:
            raise Exception("Other Tokenizer Types Not Supported Yet")

    def decode(self, data):
        print("TODO")

    def test_tokenizer(self, test_string):
        """
        Test encoding and decoding functionality of the trained tokenizer.

        Args:
            test_string (str): Sample string to encode and decode.

        Raises:
            Exception: If non-library backend or invalid text_type.
        """
        print("TESTING TOKENIZER")
        if self._text_type == "":
            raise Exception("test_string Invalid:")
        if self._tokenizer_type == "library":
            print(f"    Vocab Size: {self.tokenizer.get_vocab_size()}")
            print(f"TESTING ENCODING WITH {test_string}")
            output = self.tokenizer.encode(test_string)
            print(f"    Output IDs: {output.ids}")
            print("TESTING DECODING")
            print(f"    Decoded: {self.tokenizer.decode(output.ids)}")
        else:
            raise Exception("Other Training Tokenizer Types Not Supported Yet")
        print("TESTING TOKENIZER COMPLETE")