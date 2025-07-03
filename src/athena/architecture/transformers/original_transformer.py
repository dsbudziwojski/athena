import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self):
        print("TODO")

    # ADD other needed functionality

class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        print("TODO")

    # ADD other needed functionality

class PositionalEncoding(nn.Module):
    def __init__(self):
        print("TODO")

    # ADD other needed functionality

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