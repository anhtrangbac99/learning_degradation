import torch
from torch import nn
from basicsr.models.archs.encoder import ConvEncoder

class ConditionalEmbedding(nn.Module):
    def __init__(self, checkpoint:str,d_model:int, dim:int,device='gpu:0'):
        assert d_model % 2 == 0
        super().__init__()
        self.net = ConvEncoder()
        self.net.load_state_dict(torch.load(checkpoint))
        # self.condEmbedding = nn.Sequential(
        #     nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
        #     nn.Linear(d_model, dim),
        #     nn.SiLU(),
        #     nn.Linear(dim, dim),
        # )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.net(t)
        return emb
