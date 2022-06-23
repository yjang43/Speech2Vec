import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConditionalGRU(nn.Module):
    """Reference: https://github.com/Maluuba/gensen/blob/master/models.py
    """

    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(ConditionalGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_weights = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.hidden_weights = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.peep_weights = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()    # Following RNN intialization

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hidden, ctx):
        def recurrence(input_, hidden, ctx):
            """Recurrence helper
            """
            input_gate = self.input_weights(input_)
            hidden_gate = self.hidden_weights(hidden)
            peep_gate = self.peep_weights(ctx)
            i_r, i_i, i_n = input_gate.chunk(3, 1)
            h_r, h_i, h_n = hidden_gate.chunk(3, 1)
            p_r, p_i, p_n = peep_gate.chunk(3, 1)
            resetgate = self.sigmoid(i_r + h_r + p_r)
            inputgate = self.sigmoid(i_i + h_i + p_i)
            newgate = self.tanh(i_n + resetgate * h_n + p_n)
            # hy = newgate + inputgate * (hidden - newgate)
            hy = hidden - inputgate * (hidden - newgate)     # Revision: https://github.com/Maluuba/gensen/issues/5

            return hy
        

        output = []
        steps = range(input_.size(0))
        for i in steps:
            hidden = recurrence(input_[i], hidden, ctx)
            output.append(hidden)

        output = torch.cat(output, 0).view(input_.size(0), *output[0].size())
        return output, hidden


class Speech2Vec(nn.Module):
    """Speech2Vec implementation, inspired by Earth Species Project:
    https://github.com/earthspecies/unsupervised-speech-translation/blob/main/unsup_st/speech2vec.py
    """
    def __init__(self, input_dim=13, hidden_dim=50, window_sz=3):
        super(Speech2Vec, self).__init__()
        self.window_sz = window_sz
        
        self.encoder = nn.GRU(input_size=input_dim,
                              hidden_size=hidden_dim, 
                              bidirectional=True,
                              bias=False,
                              dropout=0)

        self.projection = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.decoders = nn.ModuleList(ConditionalGRU(input_dim=input_dim,
                                                     hidden_dim=hidden_dim,
                                                     dropout=0)
                                      for _ in range(2 * window_sz))

        self.head = nn.Linear(hidden_dim, input_dim)


    
    # center_word, context_word
    def forward(self, ctr, ctxs):
        ctr_packed = nn.utils.rnn.pack_sequence(ctr, enforce_sorted=False)
        
        _, hidden = self.encoder(ctr_packed)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        
        emb = self.projection(hidden)

        outs = []
        for k in range(2 * self.window_sz):
            # Conditional GRU, pack_sequence not supported, so pad_sequence instead.
            ctx_padded = nn.utils.rnn.pad_sequence(ctxs[k])
            out = torch.empty_like(ctx_padded)
            
            max_seq_len, batch_sz, inp_dim = ctx_padded.size()
            dec_inp = torch.zeros((1, batch_sz, inp_dim))
            hidden = emb
            steps = range(max_seq_len)
            
            for s in steps:
                _, hidden = self.decoders[k](dec_inp, hidden, emb)
                out[s] = self.head(hidden)
                dec_inp = out[s].unsqueeze(0).detach()

            # mask
            mask = torch.ones_like(out, dtype=bool)
            seq_lens = [ctx_unbatched.size(0) for ctx_unbatched in ctxs[k]]
            for i, sl in enumerate(seq_lens):
                mask[: sl, i, :] = False
            
            out.masked_fill_(mask, 0.)
            outs.append(out)
        
        return outs
    
    @torch.no_grad()
    def embed(self, ctr):
        self.eval()
        ctr_packed = nn.utils.rnn.pack_sequence(ctr, enforce_sorted=False)
        _, hidden = self.encoder(ctr_packed)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        emb = self.projection(hidden)
        
        self.train()
        return emb
