import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):
    def __init__(self, ds, source_tokenizer, target_tokenizer, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
    
        self.ds = ds
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([source_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([source_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([source_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        item = self.ds[index]['translation']
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]

        enc_input_tokens = self.source_tokenizer.encode(src_text).ids
        dec_input_tokens = self.target_tokenizer.encode(tgt_text).ids
        
        #pad sentence to seq_len
        enc_num_padding_tokens = max(0, self.seq_len - len(enc_input_tokens) - 2)  # -2 for SOS and EOS
        dec_num_padding_tokens = max(0, self.seq_len - len(dec_input_tokens) - 1)  # -1 for SOS and EOS
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too short for the given sentence pair.")
        
        #Add SOS and EOS tokens to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]   
        )
        
        #Add SOS token to the target text for decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        #Add EOS token to the target text for label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len, "Encoder input length mismatch"
        assert decoder_input.size(0) == self.seq_len, "Decoder input length mismatch"
        assert label.size(0) == self.seq_len, "Label length mismatch"
        
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #(1, 1, seq_len, seq_len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0