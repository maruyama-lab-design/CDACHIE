import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, *, d_in=128, d=128, d_hidden_factor=1, n_layers=4, hidden_dropout=0.1, residual_dropout=0.1, d_out=64):
        super().__init__()

        self.main_activation = GEGLU()
        self.last_activation = F.relu
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_hidden = int(d * d_hidden_factor)

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': nn.BatchNorm1d(d),
                        'linear0': nn.Linear(d, d_hidden*2),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = nn.LayerNorm(d)
        self.head = nn.Linear(d, d_out)

    def forward(self, x): # x: [batch, seq_len, d_model]
        x = self.first_layer(x)
        for layer in self.layers:
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x) 
        x = self.head(x)
        return x



class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)