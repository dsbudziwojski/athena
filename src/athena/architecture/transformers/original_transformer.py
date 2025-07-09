import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism used in Transformer models.

    Args:
        model_dim (int): The dimensionality of the input embeddings and model.
        num_heads (int): The number of attention heads.

    Attributes:
        Q_weights (nn.Linear): Linear transformation for computing queries.
        K_weights (nn.Linear): Linear transformation for computing keys.
        V_weights (nn.Linear): Linear transformation for computing values.
        O_weights (nn.Linear): Linear transformation for projecting concatenated attention output.
    
    Note:
        model_dim must be divisible by num_heads to ensure each head has equal dimensionality.
    """
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.Q_weights = nn.Linear(model_dim, model_dim) # Queries
        self.K_weights = nn.Linear(model_dim, model_dim) # Keys
        self.V_weights = nn.Linear(model_dim, model_dim) # Values
        self.O_weights = nn.Linear(model_dim, model_dim) # Outputs

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute the scaled dot-product attention.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, num_heads, seq_length, head_dim).
            K (Tensor): Key tensor of shape (batch_size, num_heads, seq_length, head_dim).
            V (Tensor): Value tensor of shape (batch_size, num_heads, seq_length, head_dim).
            mask (Tensor, optional): Attention mask to block certain positions.

        Returns:
            Tensor: Attention output of shape (batch_size, num_heads, seq_length, head_dim).
        """
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # K.transpose(-2, -1) swaps the last 2 columns
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)
        attention_probabilities = torch.softmax(attention_scores, dim=-1) # dim=-1 applies softmax only to last column
        attention_output = torch.matmul(attention_probabilities, V)
        return attention_output
    
    def split_heads(self, x):
        """
        Reshape the input tensor to separate multiple attention heads.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, model_dim).

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_length, head_dim).
        """
        batch_size, seq_length, model_dim = x.size()
        return x.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) # .transpose(1, 2) swaps seq_length with self.num_heads
    
    def combine_heads(self, x):
        """
        Combine multiple attention heads back into a single tensor.

        Args:
            x (Tensor): Tensor of shape (batch_size, num_heads, seq_length, head_dim).

        Returns:
            Tensor: Combined tensor of shape (batch_size, seq_length, model_dim).
        """
        batch_size, num_heads, seq_length, head_dim = x.size()
        return x.transpose(1, 2).reshape(batch_size, seq_length, self.model_dim) # .transpose(1, 2) swaps back self.num_heads with seq_length
    
    def forward(self, Q, K, V, mask=None):
        """
        Apply multi-head attention to input tensors Q, K, V.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, seq_length, model_dim).
            K (Tensor): Key tensor of shape (batch_size, seq_length, model_dim).
            V (Tensor): Value tensor of shape (batch_size, seq_length, model_dim).
            mask (Tensor, optional): Optional attention mask.

        Returns:
            Tensor: Output tensor after multi-head attention of shape (batch_size, seq_length, model_dim).
        """
        Q = self.split_heads(self.Q_weights(Q))
        K = self.split_heads(self.K_weights(K))
        V = self.split_heads(self.V_weights(V))

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.O_weights(self.combine_heads(attention_output))
        return output

class PositionalEncoding(nn.Module):
    """
    Implements fixed, sinusoidal positional encodings as introduced in the original Transformer paper.

    Args:
        model_dim (int): Dimensionality of the input embeddings.
        seq_length (int): Maximum sequence length supported.

    Attributes:
        positional_encodings (Tensor): A (1, seq_length, model_dim) tensor containing precomputed positional encodings.

    Note:
        This module uses unlearnable, deterministic encodings registered as a buffer (not updated by gradient descent).
    """
    def __init__(self, model_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        positional_encodings = torch.zeros(seq_length, model_dim)
        position_indices = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        scaling_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        
        positional_encodings[:, 0::2] = torch.sin(position_indices * scaling_term) # even indices are filled with sine
        positional_encodings[:, 1::2] = torch.cos(position_indices * scaling_term) # odd indices are filled with cosine
        self.register_buffer('positional_encodings', positional_encodings.unsqueeze(0)) # creates self.positional_encodings as an unlearnable attribute

    def forward(self, x):
        """
        Add positional encodings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, model_dim).

        Returns:
            Tensor: Positionally encoded input of the same shape.
        """
        return x + self.positional_encodings[:, :x.size(1)]
    
class LearnablePositionalEncoding(nn.Module):
    """
    Implements learnable positional encodings, allowing the model to learn optimal position representations.

    Args:
        model_dim (int): Dimensionality of the input embeddings.
        max_seq_length (int): Maximum sequence length supported by the model.

    Attributes:
        positional_encodings (nn.Parameter): A learnable tensor of shape (1, max_seq_length, model_dim).
    """
    def __init__(self, model_dim, max_seq_length):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(1, max_seq_length, model_dim)) # max_seq_length must be greater than the longest input during training

    def forward(self, x):
        """
        Add learnable positional encodings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, model_dim).

        Returns:
            Tensor: Positionally encoded input of the same shape.
        """
        return x + self.positional_encodings[:, :x.size(1), :]

class PositionWiseFeedForward(nn.Module):
    """
    Implements the position-wise feedforward network used in Transformer blocks.

    Args:
        model_dim (int): Dimensionality of input and output features.
        inner_layer_dim (int): Dimensionality of the hidden layer.

    Attributes:
        fc1 (nn.Linear): First linear transformation (expands dimensionality).
        fc2 (nn.Linear): Second linear transformation (projects back to model_dim).
        relu (nn.ReLU): Non-linear activation function.
    """
    def __init__(self, model_dim, inner_layer_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, inner_layer_dim)
        self.fc2 = nn.Linear(inner_layer_dim, model_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Apply the feedforward transformation to the input.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, model_dim).

        Returns:
            Tensor: Output tensor of the same shape after transformation.
        """
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, dimensions_model, num_heads, dimensions_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(dimensions_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(dimensions_model, dimensions_ff)
        self.norm1 = nn.LayerNorm(dimensions_model)
        self.norm2 = nn.LayerNorm(dimensions_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class DecoderLayer(nn.Module):
    def __init__(self, dimensions_model, num_heads, dimensions_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(dimensions_model, num_heads)
        self.cross_attention = MultiHeadAttention(dimensions_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(dimensions_model, dimensions_ff)
        self.norm1 = nn.LayerNorm(dimensions_model)
        self.norm2 = nn.LayerNorm(dimensions_model)
        self.norm3 = nn.LayerNorm(dimensions_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoding_output, src_mask, tgt_mask):
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))
        attention_output = self.cross_attention(x, encoding_output, encoding_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, dimensions_model, num_heads, num_layers, dimensions_ff, max_seq_len, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, dimensions_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, dimensions_model)
        self.position_encoding = PositionalEncoding(dimensions_model, max_seq_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(dimensions_model, num_heads, dimensions_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(dimensions_model, num_heads, dimensions_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(dimensions_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (src != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        no_peak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & no_peak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.position_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.position_encoding(self.decoder_embedding(tgt)))

        encoding_output = src_embedded
        for encoding_layer in self.encoder_layers:
            encoding_output = encoding_layer(encoding_output, src_mask)

        decoding_output = tgt_embedded
        for decoding_layer in self.decoder_layers:
            decoding_output = decoding_layer(decoding_output, encoding_output, src_mask, tgt_mask)

        output = self.fc(decoding_output)
        return output