import torch
import numpy as np
from torch import nn

# in_channel
n=2
#mid_channel
nf=64

##H (2*32*64)as the input of discriminator
class Discriminatior(nn.Module):
    def __init__(self):
        super(Discriminatior,self).__init__()

        def encoder_block(in_feat, out_feat):
            layer = []
            layer.append(nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1))  # channwl*W*H->(2channel)*(W/2)*(H/2))
            layer.append(nn.BatchNorm2d(out_feat, 0.8))
            layer.append(nn.LeakyReLU(0.2, inplace=True))
            return layer
        #2*32*64->1024*1*2
        self.encoder=nn.Sequential(
            *encoder_block(n,nf),
            *encoder_block(nf,nf*2),
            *encoder_block(nf*2,nf*4),
            *encoder_block(nf*4,nf*8),
            *encoder_block(nf*8,nf*16)
        )
        #1024*1*2->1*1*1
        self.conv=nn.Conv2d(in_channels=nf*16,out_channels=1,kernel_size=(3,4),stride=1,padding=1)
        self.sig=nn.Sigmoid()


    def forward(self,x):
        x=self.encoder(x)
        x=self.conv(x)
        x=self.sig(x)
        return x

discriminator=Discriminatior()
print(discriminator)

torch.save(discriminator,'./models/discriminator.pth')
print('判别网络保存成功!')

