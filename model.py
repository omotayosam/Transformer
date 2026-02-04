import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
       super().__init__()
       self.d_model = d_model
       self.seq_len = seq_len
       self.dropout = nn.Dropout(dropout)
       
       #matrix of (seq_len, d_model)
       pe = torch.zeros(seq_len, d_model)
       
       #vector of shape(seq_len, 1)
       position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
       
       #applying mathematical identity a^{x} = e^x ln(a)}
       #therefore 1000^(2i/d_model) = e^(2i/d_model) ln(10000)
       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
       
       #Applying sin & cos to even & odd positions respectively
       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)
       
       #since pe is initially a matrix of (seq_len, d_model) to make the dimensions/shape match, we broadcast
       pe = pe.unsqueeze(0) # from (seq_len, d_model) -> (1, seq_len, d_model)
       
       self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
       
class LayerNormalization(nn.Module):
    """ Layer norm is a variation of the z-score from statistics, applied to a single vec-
        tor in a hidden layer. µ = mean, σ = standard deviation 
    """
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        #2 learnable parameters, γ and β , representing gain and offset values, are introduced
        #layerNorm(x) = γ * ((x - mean)/standar_deviation) + β
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added
        #ε (epsilon) is added to guard against numerical failure
        
        
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        #putting the formula together LayerNorm(x) = γ * ((x - mean)/standard_deviation) + β
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.h = h
        assert d_model % h == 0, "dmodel is not divisible by h"
        
        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model) #Wq
        self.W_k = nn.Linear(d_model, d_model) #Wk
        self.W_v = nn.Linear(d_model, d_model) #Wv
        
        self.W_o = nn.Linear(d_model, d_model) #Wo
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        numerator = torch.matmul(query, key.transpose(-2, -1)) # (Batch, h, Seq_len, d_k) x (Batch, h, d_k, Seq_len) ---> (Batch, h, Seq_len, Seq_len)
        denominator = math.sqrt(d_k)
        scores = numerator / denominator
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = torch.softmax(scores, dim = -1) # (Batch, h, Seq_len, Seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        output = torch.matmul(attention_scores, value) # (Batch, h, Seq_len, Seq_len) x (Batch, h, Seq_len, d_k) ---> (Batch, h, Seq_len, d_k)
        return output, attention_scores


    def forward(self, q, k, v, mask):
        query = self.W_q(q) # (batch_size, seq_len, d_model) ---> (batch_size, seq_len, d_model)
        key = self.W_k(k) # (batch_size, seq_len, d_model) ---> (batch_size, seq_len, d_model)
        value = self.W_v(v) # (batch_size, seq_len, d_model) ---> (batch_size, seq_len, d_model)

        # (Batch, Seq_len, d_model) ---> (Batch, Seq_len, h, d_k) ---> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.W_o(x) # (Batch, Seq_len, d_model) ---> (Batch, Seq_len, d_model) h * d_k) ---> (Batch, Seq_len, d_model)
    
    
class ResidualConnectionBlock(nn.Module):
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.layer_norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

class EncoderBlock (nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class LanguageModelingHead(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.lang_model_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.lang_model_head(x), dim=-1)
    
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, src_position: PositionalEncoding, 
                 tgt_position: PositionalEncoding, language_model_head: LanguageModelingHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.language_model_head = language_model_head
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_position(self.src_embed(src)), src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        #tgt = self.tgt_embed(tgt)
        #tgt = self.tgt_position(tgt)
        return self.decoder(self.tgt_position(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)
    
    def head(self, x):
        return self.language_model_head(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, 
                      d_model: int = 512, N: int = 6, h: int = 8, 
                      dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    
    #create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    #create positional encoding layers
    src_position = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_position = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    #Building Encoder
    encoder_layers = nn.ModuleList([EncoderBlock(
        MultiHeadAttentionBlock(d_model, h, dropout),
        FeedForwardBlock(d_model, d_ff, dropout),
        dropout) for _ in range(N)])
    encoder = Encoder(encoder_layers)
    
    #Building Decoder
    decoder_layers = nn.ModuleList([DecoderBlock(
        MultiHeadAttentionBlock(d_model, h, dropout),
        MultiHeadAttentionBlock(d_model, h, dropout),
        FeedForwardBlock(d_model, d_ff, dropout),
        dropout) for _ in range(N)])
    decoder = Decoder(decoder_layers)
    
    #create the head
    language_model_head = LanguageModelingHead(d_model, tgt_vocab_size)
    
    #create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_position, tgt_position, language_model_head)
    
    #initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer