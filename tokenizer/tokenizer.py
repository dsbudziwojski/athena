from tokenizers import Tokenizer, normalizers, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents


class GeneralTokenizer(object):
    """
    A flexible tokenizer interface that supports both a HuggingFace Tokenizers library-based BPE
    implementation (when tokenizer_type='library') and a custom implementation stub (when
    tokenizer_type='default').

    Attributes:
        _text_type (str): Input data format ('list' or 'file').
        _tokenizer_type (str): Backend choice ('default' for custom, 'library' for BPE via HF).
        _model_type (str): Model architecture ('BPE').
        tokenizer (Tokenizer): HuggingFace Tokenizer instance when using library backend.
        _trainer (BpeTrainer): Trainer for list-based BPE training (library only).
    """
    def __init__(self, text_type="file", tokenizer_type="library", model_type="BPE"):
        """
        Initialize the GeneralTokenizer.

        Args:
            text_type (str): Data source type ('list' for in-memory lists, 'file' for file-based input).
            tokenizer_type (str): Which backend to use ('library' for HF BPE or 'default' as a stub).
            model_type (str): Model architecture ('BPE' supported).

        Raises:
            Exception: If tokenizer_type or model_type is unsupported.
        """
        if tokenizer_type not in ["default", "library"]:
            raise Exception("Invalid tokenizer")
        if model_type not in ["BPE"]:
            raise Exception("Invalid model")

        self._text_type = text_type
        self._tokenizer_type = tokenizer_type
        self._model_type = model_type

        if tokenizer_type == "library":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            if text_type == "list":
                self._trainer = trainers.BpeTrainer(
                    vocab_size=30_000,
                    min_frequency=2,
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                )
            elif text_type == "file":
                raise Exception("text_type Error: file not yet supported")
            else:
                raise Exception("Invalid text_type")

    def train_tokenizer(self, data):
        """
        Train the tokenizer on provided data using the configured backend.

        Args:
            data (Iterable[str]): List of text strings for training when text_type='list'.

        Raises:
            Exception: If file-based training is selected (not supported) or backend stub.
        """
        print("TRAINING TOKENIZER STARTING")
        if self._tokenizer_type == "library":
            if self._text_type == "list":
                self.tokenizer.train_from_iterator(data, self._trainer)
            elif self._text_type == "file":
                raise Exception("text_type Error: file not yet supported")
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
            Exception: If backend stub or invalid text_type.
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