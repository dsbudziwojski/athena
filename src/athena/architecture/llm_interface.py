import torch
import torch.nn as nn
import torch.optim as optim
from transformers.original_transformer import Transformer

# Model Evaluation & Visualization
def evaluate_llm(visualize=True):
    # Load the saved model
    transformer = Transformer()
    file_path = 'src/athena/architecture/transformer_model.pth'
    transformer.load_state_dict(torch.load(file_path))

    transformer.eval() # set the model to evaluation mode, disabling dropout and batch normalization
    if visualize:
        print("FANCY PLOT")

# Training Function
def train_loop():
    # Initialize transformer and its hyperparameters
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    model_dim = 512
    num_heads = 8
    num_layers = 6
    dimensions_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    transformer = Transformer(src_vocab_size, tgt_vocab_size, model_dim, num_heads, num_layers, dimensions_ff, max_seq_length, dropout)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    
    # Training Loop
    criterion = nn.CrossEntropyLoss(ignore_index=0) # loss will not consider the first index because its reserved for padding tokens
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    transformer.train() # sets the model to training mode, enabling dropout and batch normalization
    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1]) # passes the source and target data through the transformer, target data is shifted by 1 one token
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1)) # calculates the loss through CrossEntropyLoss, reshapes data into 1-D tensors
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    
    # Save the model
    file_path = 'src/athena/architecture/transformer_model.pth'
    torch.save(transformer.state_dict(), file_path)

# Interface
class AthenaLLM():
    def __init__(self, model_type, wikipedia_only = True, auto_setup=True):  # params here are superset up _manual_setup
        print("TODO: user defines what they want")
        # be able to provides weights & hyperparams...
        if wikipedia_only:
            print("TODO")
        if auto_setup:
            self._manual_setup()

    def add_data(self, path):
        print("TODO: manually allow data to be added in addition to wikipedia")

    def _manual_setup(self):
        print("TODO: user begins setup (e.g. training, so forth...)")

    def save_weights(self):
        print("TODO: save weights/hyperparams")

    def prompt(self):
        print("TODO: user can use")

def main():
    train_loop()
    evaluate_llm()

if __name__ == "__main__":
    main()