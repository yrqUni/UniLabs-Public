import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import random

class Masker:
    def __init__(self, mask_rate, mask_val='*'):
        self.mask_rate = mask_rate
        self.mask_val = mask_val
    def gen_data(self, input_str):
        mask_positions_indices = random.sample(range(len(input_str)), int(len(input_str) * self.mask_rate))
        masked_sample = ''.join([self.mask_val if i in mask_positions_indices else char for i, char in enumerate(input_str)])
        masked_sample = masked_sample
        mask_positions = ''.join(['1' if i in mask_positions_indices else '0' for i in range(len(input_str))])
        raw_sample = input_str
        return {'mask_positions':mask_positions, 'masked_sample':masked_sample, 'raw_sample':raw_sample}

class SimpleTokenizer:
    def __init__(self, dict_path):
        self.char_to_index = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f, start=1):  
                char = line.strip()  
                self.char_to_index[char] = index
    def tokenize(self, seq):
        return [self.char_to_index.get(char, 0) for char in seq]

class FastaDataset(Dataset):
    def __init__(self, filepath, seq_len, dict_path, mask_rate):
        self.filepath = filepath
        self.seq_len = seq_len 
        self.tokenizer = SimpleTokenizer(dict_path)
        self.records = list(SeqIO.parse(self.filepath, "fasta"))
        self.PAD = '='
        self.CLS = '@'
        self.MASK = '*'
        self.masker = Masker(mask_rate, self.MASK)
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        if len(str(self.records[idx].seq)) > (self.seq_len-1):
            seq = str(self.records[idx].seq)[:self.seq_len-1]
            len_rawseq = len(seq)
        else: 
            seq = str(self.records[idx].seq)
            len_rawseq = len(seq)
        seq = self.masker.gen_data(seq)
        seq['masked_sample'] = seq['masked_sample'] + self.PAD * (self.seq_len - len(seq['masked_sample']) - 1)
        seq['masked_sample'] = seq['masked_sample'] + self.CLS
        seq['raw_sample'] = seq['raw_sample'] + self.PAD * (self.seq_len - len(seq['raw_sample']) - 1)
        seq['raw_sample'] = seq['raw_sample'] + self.CLS
        seq['mask_positions'] = seq['mask_positions'] + '0' * (self.seq_len - len(seq['mask_positions']))
        tokenized_masked_seq = self.tokenizer.tokenize(seq['masked_sample'])
        tokenized_raw_seq = self.tokenizer.tokenize(seq['raw_sample'])
        return {'masked_seq': torch.tensor(tokenized_masked_seq),
               'raw_seq': torch.tensor(tokenized_raw_seq),
               'mask_pos': torch.tensor([int(i) for i in list(seq['mask_positions'])])}