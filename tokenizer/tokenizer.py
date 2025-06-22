print("tokenizer")
from tokenizers import Tokenizer, normalizers, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents

def tokenize(data, text_type="file", tokenizer="default", model="BPE"):
    if tokenizer not in ["default", "library"]:
        raise Exception("Invalid tokenizer")
    if model not in ["BPE"]:
        raise Exception("Invalid model")

    if tokenizer == "library":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        if text_type == "list":
            trainer = trainers.BpeTrainer(
                vocab_size=30_000,
                min_frequency=2,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            )
            tokenizer.train_from_iterator(data, trainer)
            print("Training of Tokenizer from iterator Complete")
            print(tokenizer.get_vocab_size())
            print("Testing encoding")
            output = tokenizer.encode("Python is family!")
            print(output.ids)
            print("Testing decoding")
            print(tokenizer.decode(output.ids))
            print("TESTING DONE")
        elif tokenizer == "file":
            print("TODO")
        else:
            raise Exception("Invalid text_type")

    else:
        # our implementation
        print("TODO")
