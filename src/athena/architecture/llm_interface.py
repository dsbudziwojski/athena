import torch
import torch.nn as nn
import torch.optim as optim
from athena.architecture.transformers.original_transformer import Transformer as OG_Transformer
from athena.tokenizer.tokenizer import GeneralTokenizer
from athena.scraper import Scraper
from utilities import src_trg_json_spliter

# Model Evaluation & Visualization
def evaluate_llm(visualize=True):
    # Load the saved model
    transformer = OG_Transformer #
    file_path = 'src/athena/architecture/transformer_model.pth'
    transformer.load_state_dict(torch.load(file_path))

    transformer.eval() # set the model to evaluation mode, disabling dropout and batch normalization
    if visualize:
        print("FANCY PLOT")

# Training Function
def train_loop(transformer, src_data, tgt_data, src_vocab_size, tgt_vocab_size, lr=0.001, betas=(0.9, 0.98), eps=1e-9):
    # Training Loop
    # why do i need to turn this into torch.tensor(...)
    src_data = torch.tensor(src_data, dtype=torch.long)
    tgt_data = torch.tensor(tgt_data, dtype=torch.long)
    # print(type(tgt_data))
    # print(isinstance(tgt_data, list), isinstance(tgt_data, torch.Tensor))
    criterion = nn.CrossEntropyLoss(ignore_index=0) # loss will not consider the first index because its reserved for padding tokens
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=betas, eps=eps)
    transformer.train() # sets the model to training mode, enabling dropout and batch normalization
    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1]) # passes the source and target data through the transformer, target data is shifted by 1 one token
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1)) # calculates the loss through CrossEntropyLoss, reshapes data into 1-D tensors
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Interface
class AthenaLLM():
    def __init__(self, model_type="original", tokenizer_type="library", wikipedia_only=True, model_dim=512, num_heads=8,
                 num_layers=6, dimensions_ff=2048, max_seq_len=128, dropout=0.1, auto_train_tokenizer=True, auto_train_model=True):
        print("INSTANTIATING athenaLLM")
        print("    CHECKING BASIC VALIDITY OF ARGUMENTS: ", end="")
        if auto_train_model and not auto_train_tokenizer:
            raise Exception("Invalid Argument: can not have auto_train_model without not auto_train_tokenizer")
        print("SUCCESSFUL")
        print("    SAVING ARGUMENTS FOR FUTURE USE: ", end="")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dimensions_ff = dimensions_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        print("SUCCESSFUL")
        print("    CREATING TOKENIZER: ", end="")
        if tokenizer_type == "library":
            self.tokenizer = GeneralTokenizer(text_type="files")
        else:
            raise Exception("Invalid Tokenizer")
        print("SUCCESSFUL")
        if auto_train_tokenizer:
            print("    FETCHING REQUESTED DATA: ", end="")
            if wikipedia_only:
                self.scraper = Scraper(["https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles18.xml-p26716198p27121850.bz2"])
                data_files = self.scraper.get_data_files()
                self._src_data, self._trg_data = src_trg_json_spliter(data_files[0]) # write this into a file instead
            else:
                raise Exception("Invalid Data Request: currently not supported")
            print("SUCCESSFUL")
            print("    TRAINING TOKENIZER: ", end="")
            self.tokenizer.train_tokenizer(["dump_data.json"]) # add data here... like this in the future: ["dump_data.json"]
            self.src_vocab_size = self.tgt_vocab_size = self.tokenizer.get_vocab_size()
            print("SUCCESSFUL")
        print("    TURNING SRC AND TGT INTO LISTS OF ENCODINGS: ", end="")
        vectorized_src_ids = self.tokenizer.encode(self._src_data)
        vectorized_tgt_ids = self.tokenizer.encode(self._trg_data)
        print("SUCCESSFUL")
        print("    CREATING TRANSFORMER: ", end="")
        if model_type == "original":
            self.transformer = OG_Transformer(
                self.src_vocab_size,
                self.tgt_vocab_size,
                self.model_dim,
                self.num_heads,
                self.num_layers,
                self.dimensions_ff,
                self.max_seq_len, # GOT TO FIX THIS... ADD PADDING ETC TO THE STUFF... # half sentence bc of truncation??? does work?
                self.dropout
            )
        else:
            raise Exception("Invalid Transformer")
        print("SUCCESSFUL")
        if auto_train_model:
            print("    BEGINNING AUTO-TRAINING \"GODDESS OF WISDOM\": ")
            train_loop(self.transformer, vectorized_src_ids, vectorized_tgt_ids, self.src_vocab_size, self.tgt_vocab_size)
            print("SUCCESSFUL")
        else:
            print("    \"GODDESS OF WISDOM\" IS READY TO BE TRAINED BY USER!")


    def add_data(self, path):
        print("TODO: manually allow data to be added in addition to wikipedia")

    def _manual_setup(self):
        print("TODO: user begins setup (e.g. training, so forth...)")

    def save_weights(self):
        # Save the model
        file_path = 'src/athena/architecture/transformer_model.pth'
        torch.save(self.transformer.state_dict(), file_path)

    def prompt(self):
        print("TODO: user can use")

def main():
    athena = AthenaLLM()
    athena.save_weights()

if __name__ == "__main__":
    main()