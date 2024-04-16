import torch
import torch.nn as nn
from mamba import Mamba, MambaConfig

class dec_mlp(nn.Module):
    def __init__(self, d_model=768, out_fea=1, mlp_droprate=0.1):
        super(dec_mlp, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_model)
        self.fc2 = torch.nn.Linear(d_model, out_fea)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(mlp_droprate)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class cls_mlp(nn.Module):
    def __init__(self, d_model=768, out_fea=1, mlp_droprate=0.1):
        super(cls_mlp, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_model)
        self.fc2 = torch.nn.Linear(d_model, out_fea)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(mlp_droprate)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class protmamba(nn.Module):
    def __init__(self, mode='pt', seqlen=1000, vocab=28,
                 n_layers=2, d_model=768, expand_factor=2, d_state=16, 
                 pscan=True, num_cls=2, cls_mlp_droprate=0.1, dec_mlp_droprate=0.1):
        super(protmamba, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, seqlen, d_model))
        mamba_config = MambaConfig(d_model=d_model, n_layers=n_layers,
                                   pscan=pscan, expand_factor=expand_factor, 
                                   d_state=d_state,)
        self.mamba = Mamba(mamba_config)
        self.cls_mlp_block = cls_mlp(d_model=d_model, 
                                     out_fea=num_cls, 
                                     mlp_droprate=cls_mlp_droprate)
        self.dec_mlp_block = dec_mlp(d_model=d_model, 
                                     out_fea=vocab, 
                                     mlp_droprate=dec_mlp_droprate)
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_enc
        x = self.mamba(x)
        x_cls = self.cls_mlp_block(x[:,-1,:])
        x = self.dec_mlp_block(x)
        return x, x_cls
        