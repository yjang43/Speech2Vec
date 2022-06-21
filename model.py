import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



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

        self.reset_parameters()    # Following RNN intialization

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden, ctx):
        def recurrence(input, hidden, ctx):
            """Recurrence helper
            """
            input_gate = self.input_weights(input)
            hidden_gate = self.hidden_weights(hidden)
            peep_gate = self.peep_weights(ctx)
            i_r, i_i, i_n = input_gate.chunk(3, 1)
            h_r, h_i, h_n = hidden_gate.chunk(3, 1)
            p_r, p_i, p_n = peep_gate.chunk(3, 1)
            resetgate = F.sigmoid(i_r + h_r + p_r)
            inputgate = F.sigmoid(i_i + h_i + p_i)
            newgate = F.tanh(i_n + resetgate * h_n + p_n)
            # hy = newgate + inputgate * (hidden - newgate)
            hy = hidden - inputgate * (hidden - newgate)     # Revision: https://github.com/Maluuba/gensen/issues/5

            return hy
        

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, ctx)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
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


    def forward(self, x_n, xs_k):
        x_n = nn.utils.rnn.pack_sequence(x_n, enforce_sorted=False)
        
        _, h_n = self.encoder(x_n)
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        
        z_n = self.projection(h_n)

        ys_k = []            
        for k in range(2 * self.window_sz):
            # Conditional GRU, pack_sequence not supported, so pad_sequence instead.
            # Note that bias is zero for decoder.
            x_k = nn.utils.rnn.pad_sequence(xs_k[k])
            out_k, _ = self.decoders[k](x_k[: -1], z_n, z_n)
            y_k = self.head(out_k)
            
            ys_k.append(y_k)
        
        return ys_k
    
    
    def embed(self, x_n):
        self.eval()
        x_n = nn.utils.rnn.pack_sequence(x_n, enforce_sorted=False)
        _, h_n = self.encoder(x_n)
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        z_n = self.projection(h_n)
        
        self.train()
        return z_n
            