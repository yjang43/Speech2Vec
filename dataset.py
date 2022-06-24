import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LibriSpeechDataset(Dataset):
    """Module for LibriSpeech datset. Current implementation suffers from IO overhead
    """
    
    def __init__(self, data_dir, window_sz=3):
        self.data_dir = data_dir
        self.window_sz = window_sz
        self.mfcc = transforms.MFCC(
            sample_rate=16000,
            n_mfcc=13, 
            log_mels=False, 
            melkwargs={
                'n_fft': 400,       # 0.025s
                'hop_length': 160,  # 0.01s
                'n_mels': 13,        # keep filter bank the same as n_mfcc
            }
        )
        
        # load words and index
        with open(os.path.join(data_dir, 'words.txt'), 'r') as f:
            self.words = f.read().split()
        with open(os.path.join(data_dir, 'index.txt'), 'r') as f:
            self.index = f.read().split()
    
        assert len(self.words) == len(self.index)
    
    def __len__(self):
        return len(self.words) - 2 * self.window_sz
    
    def __getitem__(self, idx):
        
        fps = [fp + '.wav' for fp in self.index[idx: idx + 2 * self.window_sz + 1]]
        ds = []
        for fp in fps:
            d, _= torchaudio.load(os.path.join(self.data_dir, *fp.split('-')))
            d = self.mfcc(d).squeeze(0).transpose(0, 1)
            ds.append(d)
        ws = self.words[idx: idx + 2 * self.window_sz + 1]
        
        src, tgts = ds[self.window_sz], ds[: self.window_sz] + ds[self.window_sz + 1: ]
        src_word, tgt_words = ws[self.window_sz], ws[: self.window_sz] + ws[self.window_sz + 1: ]
        
        
        return {'src': src, 'tgts': tgts, 
                'src_word': src_word, 'tgt_words': tgt_words}
    

    # collate function
    def collate_fn(self, batch):
        src = [item['src'] for item in batch]
        # src = nn.utils.rnn.pack_sequence(src, enforce_sorted=False)
        # src = nn.utils.rnn.pad_sequence(src)
        src_word = [item['src_word'] for item in batch]

        tgts = []
        tgt_words = []
        for i in range(2 * self.window_sz):
            tgt = [item['tgts'][i] for item in batch]
            # tgt = nn.utils.rnn.pack_sequence(tgt, enforce_sorted=False)
            # tgt = nn.utils.rnn.pad_sequence(tgt)
            tgt_word = [item['tgt_words'][i] for item in batch]
            
            tgts.append(tgt)
            tgt_words.append(tgt_word)


        return src, tgts, src_word, tgt_words