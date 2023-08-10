import torch
from torch import nn

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int,device = 'cuda:0'):
        assert d_model % 2 == 0
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,kernel_size=1)
        self.SiLU = nn.SiLU()
        self.conv2 = nn.Conv2d(1,1,kernel_size=1)

        self.conv0 = nn.Conv2d(3,1,kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(1)

        # self.linear0 = nn.Linear(int(dim/2),dim) 
        self.linear = nn.Linear(dim,1)
    def forward(self, t:torch.Tensor) -> torch.Tensor:
        B,_,H,W = t.shape
        t_0 = self.SiLU(self.batch_norm1(self.conv0(t)))
        # t = torch.fft.fft2(t_0,norm='backward')
        # t_image = t.imag
        # t_real = t.real 
        # t_f = torch.cat([t_real,t_image],dim=1)
        # .type(torch.FloatTensor)
        emb = self.conv1(t_0)
        # print(emb.shape)
        # emb = self.pool1(emb)
        emb = self.SiLU(emb)
        emb = self.conv2(emb)
        # t_real,t_imag = torch.chunk(emb,2,dim=1)
        # emb = torch.complex(t_real,t_imag)
        # emb = torch.fft.ifft2(emb,s=(H,W),norm='backward').type(torch.FloatTensor)
        # emb = emb.cuda()
        # emb = self.pool2(emb)
        # emb = self.linear0(emb)
        return self.linear(emb).view(B,-1)


# class ConditionalEmbedding(nn.Module):
#     def __init__(self, num_labels:int, d_model:int, dim:int):
#         assert d_model % 2 == 0
#         super().__init__()
#         self.condEmbedding = nn.Sequential(
#             nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
#             nn.Linear(d_model, dim),
#             nn.SiLU(),
#             nn.Linear(dim, dim),
#         )

#     def forward(self, t:torch.Tensor) -> torch.Tensor:
#         emb = self.condEmbedding(t)
#         return emb
  