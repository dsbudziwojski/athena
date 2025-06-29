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
            if text_type == "list" or text_type == "files":
                self._trainer = trainers.BpeTrainer(
                    vocab_size=30_000,
                    min_frequency=2,
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                )
            else:
                raise Exception("Invalid text_type")

    def train_tokenizer(self, data):
        """
        Train the tokenizer on provided data using the configured backend.

        Args:
            data (Iterable[str]): Text strings if text_type='list', or file paths if text_type='files'.

        Raises:
            Exception: If invalid text_type or non-library backend.
        """
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
        print("TRAINING TOKENIZER COMPLETE")
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