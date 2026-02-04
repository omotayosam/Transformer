import torch
import torch.nn as nn
import warnings

from tqdm import tqdm

from config import get_weights_file_path, get_config
from dataset import BilingualDataset, causal_mask 
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from model import build_transformer

def greedy_decode(model, source, source_mask, source_tokenizer, target_tokenizer,max_len, device):
    sos_idx = target_tokenizer.token_to_id('[SOS]')
    eos_idx = target_tokenizer.token_to_id('[EOS]')
    
    # Precompute encoder output and reuse it for every token gotten from decoder
    encoder_output = model.encode(source, source_mask)
    #Initialize the target sequence with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)  #shape: (1, 1)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # Build the target mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)  #shape: (1, seq_len, seq_len)
        
        #Get the decoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)  #shape: (1, seq_len, d_model)
        
        #Get the next token probabilities
        prob = model.head(out[:, -1])  #shape: (1, target_vocab_size)
        # Select the token with the highest probability
        _, next_word = torch.max(prob, dim = 1)  #shape: (1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)  #shape: (1, seq_len + 1)
        
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)
        
        
def run_validation(model, validation_ds, source_tokenizer, target_tokenizer, max_len, device, print_msg, global_state, writer, num_examples = 2):
    model.eval()
    count = 0
    
    # source_texts = []
    # excepted = []
    # predicted = []
    
    #Size of the control window (just use a batch default value)
    console_width = 80
    
    if num_examples > 0:
        with torch.no_grad():
            for batch in validation_ds:
                encoder_input = batch['encoder_input'].to(device) #shape: (1, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) #shape: (1, 1, 1, seq_len)
                
                assert encoder_input.size(0) == 1, "Batch size should be 1 for validation"
                
                model_output = greedy_decode(model, encoder_input, encoder_mask, source_tokenizer, target_tokenizer, max_len, device)
                
                source_text = batch['src_text'][0]
                target_text = batch['tgt_text'][0]
                model_output_text = target_tokenizer.decode(model_output.detach().cpu().numpy())
                
                # source_texts.append(source_text)
                # excepted.append(target_text)
                # predicted.append(model_output_text)
                
                # print some examples to the console
                
                print(f"{'-'*console_width}")
                print(f"SOURCE: {source_text}")
                print(f"EXPECTED: {target_text}")
                print(f"PREDICTED: {model_output_text}")
                print(f"{'-'*console_width}")
                
                count += 1
                if count == num_examples:
                    break
                    

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                                   min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["source_lang"]}-{config["target_lang"]}', split='train')
    
    # Build or load tokenizers
    source_tokenizer = get_or_build_tokenizer(config, ds_raw, config['source_lang'])
    target_tokenizer = get_or_build_tokenizer(config, ds_raw, config['target_lang'])
    
    #split dataset into train and validation sets (90% train, 10% val)
    ds_train_size = int(0.9 * len(ds_raw))
    ds_val_size = len(ds_raw) - ds_train_size
    ds_train_raw, ds_val_raw = random_split(ds_raw, [ds_train_size, ds_val_size])
    
    train_ds = BilingualDataset(ds_train_raw, source_tokenizer, target_tokenizer, 
                                config['source_lang'], config['target_lang'], config['seq_len'])
    val_ds = BilingualDataset(ds_val_raw, source_tokenizer, target_tokenizer, 
                                config['source_lang'], config['target_lang'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = source_tokenizer.encode(item['translation'][config['source_lang']]).ids
        tgt_ids = target_tokenizer.encode(item['translation'][config['target_lang']]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, source_tokenizer, target_tokenizer


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, source_tokenizer, target_tokenizer = get_ds(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
    
    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=target_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device) #label smoothing to reduce overfitting
    
    for epoch in range(initial_epoch, config['num_epochs']):
        #model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in batch_iterator:
            model.train()
            
            encoder_input = batch['encoder_input'].to(device) #shape: (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) #shape: (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #shape: (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # shape: (Batch, 1, seq_len, seq_len)
            
            # Run the tensors through the transformers
            encoder_output = model.encode(encoder_input, encoder_mask) # shape: (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # shape: (Batch_size, seq_len, d_model)
            proj_output = model.head(decoder_output) #shape: (batch_size, seq_len, target_vocab_size)
            
            label = batch['label'].to(device) #shape: (batch_size, seq_len)
            
            # shape (B, seq_len, tgt_vocab_size) --> (B * Seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            #log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            #Backpropagate the loss
            loss.backward()
            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        #run validation at the end of each epoch
        run_validation(model, val_dataloader, source_tokenizer, target_tokenizer, 
                           config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_state=global_step, writer=writer, num_examples=2)
            
        # save the model at the end of each epoch
        model_filename = get_weights_file_path(config, f'{epoch: 02d}')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)