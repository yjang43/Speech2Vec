import os
import numpy as np
import torch
import torch.nn as nn

from bisect import bisect_right
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from python_speech_features import mfcc


class LibriSpeechDataset(Dataset):
    """Module for LibriSpeech datset. Current implementation suffers from IO overhead
    """
    
    def __init__(self, data_dir, word_dir, window_sz=3):
        self.data_dir = data_dir
        self.window_sz = window_sz
        self.npzs = sorted([fn for fn in os.listdir(data_dir) if fn[-4:] == '.npz'])
        txts = sorted([fn for fn in os.listdir(word_dir) if fn[-4:] == '.txt'])
        self.ptrs = [0]
        self.words = []
                
        ptr_cnt = 0

        npz_progbar = tqdm(self.npzs)
        txt_progbar = tqdm(txts)
        for npz in npz_progbar:
            word_cnt = np.load(os.path.join(data_dir, npz))['word_cnt']
            ptr_cnt += word_cnt
            self.ptrs.append(ptr_cnt)
            # npz_progbar.set_description(f"file: {npz}")
            # npz_progbar.refresh()
        for txt in txt_progbar:
            with open(os.path.join(word_dir, txt), 'r') as f:
                ws = f.read()
                self.words += ws.split()
            # txt_progbar.set_description(f"file: {txt}")
            # txt_progbar.refresh()
    
    def __len__(self):
        return self.ptrs[-1] - 2 * self.window_sz
    
    def __getitem__(self, idx):
        ni = bisect_right(self.ptrs, idx)    # locate npz file
        npz = self.npzs[ni - 1]
        data = np.load(os.path.join(self.data_dir, npz))
        
        ds = []
        ws = []
        
        for k in range(2 * self.window_sz + 1):
            # load next npz file if necessary
            if self.ptrs[ni] <= idx + k:
                ni = ni + 1
                npz = self.npzs[ni - 1]
                data = np.load(os.path.join(self.data_dir, npz))
            
            d = data[f'arr_{idx + k - self.ptrs[ni - 1]}']
            d = mfcc(d, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13)
            d = torch.from_numpy(d).float()
            w = self.words[idx + k]
            
            ds.append(d)
            ws.append(w)
            
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

    

class LibriSpeechDatasetFast(Dataset):
    """This implementation loads data in memory to avoid IO overhead.
    """
    def __init__(self, data_dir, word_dir, window_sz=3):
        self.data_dir = data_dir
        self.window_sz = window_sz
        self.npzs = sorted([fn for fn in os.listdir(data_dir) if fn[-4:] == '.npz'])
        txts = sorted([fn for fn in os.listdir(word_dir) if fn[-4:] == '.txt'])
        
        self.datas = []
        self.words = []

        npz_progbar = tqdm(self.npzs)
        txt_progbar = tqdm(txts)
        for npz in npz_progbar:
            data = np.load(os.path.join(data_dir, npz))
            word_cnt = data['word_cnt']
            for i in range(word_cnt):
                d = data[f'arr_{i}']
                d = mfcc(d, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13)
                d = torch.from_numpy(d).float()
                self.datas.append(d)
            # npz_progbar.set_description(f"file: {npz}")
            # npz_progbar.refresh()
        for txt in txt_progbar:
            with open(os.path.join(word_dir, txt), 'r') as f:
                ws = f.read()
                self.words += ws.split()
            # txt_progbar.set_description(f"file: {txt}")
            # txt_progbar.refresh()
    
    def __len__(self):
        return len(self.words) - 2 * self.window_sz
    
    def __getitem__(self, idx):

        ds = self.datas[idx: idx + 2 * self.window_sz + 1]
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
