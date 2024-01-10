import torch
import torch.nn as nn
class SelfAttetion(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttetion,self).__init__
        self.embed_size=embed_size
        self.heads=heads
        self.head_dim=embed_size//heads
        assert(self.head_dim*heads==embed_size),"Embed size need to be div by heads"

        self.values=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.fc_out=nn.Linear(heads*self.head_dim,embed_size)
    def forward(self,values,keys,query,mask):
        N=query.shape[0]
        value_len,key_len,query_len=values.shape[1],values.shape[1],values.shape[1]

        #Split embedding into self.heads pieces
        values=values.reshape(N.value_len,self.heads,self.head_dim)
        keys=keys.reshape(N.value_len,self.heads,self.head_dim)
        queries=queries.reshape(N.value_len,self.heads,self.head_dim)

        energy=torch.einsum("nghd,nkh->nhqk")