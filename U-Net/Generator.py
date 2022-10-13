import torch
import numpy as py
from torch import nn
from torch.nn import functional as F
# from torch.utils.tensorboard import SummaryWriter
import h5py

# writer=SummaryWriter(log_dir='logs')
#in_channel
n=2

#mid_channel
nf=64


#construct a generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.ConvT= nn.ConvTranspose2d(in_channels=n,out_channels=n,kernel_size=(4,1),stride=(4,1))
        self.Conv1=nn.Conv2d(in_channels=n,out_channels=n,kernel_size=3,stride=1,padding=1)
        self.LeakyRELU=nn.LeakyReLU(0.2)
        def encoder_block(in_feat,out_feat):
            layer=[]
            layer.append(nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1))# channwl*W*H->(2channel)*(W/2)*(H/2))
            layer.append(nn.BatchNorm2d(out_feat, 0.8))
            layer.append( nn.LeakyReLU(0.2, inplace=True))
            return layer

        def decoder_block(in_feat,out_feat):
            layer=[]
            #upsampling:channel->channel/2,W*H->2W*2H
            layer.append(nn.ConvTranspose2d(in_channels=in_feat,out_channels=out_feat,kernel_size=4,stride=2,padding=1))
            layer.append(nn.BatchNorm2d(out_feat,0.8))
            layer.append(nn.LeakyReLU(0.2,inplace=True))
            return layer

        ## 3 encoder
        self.encoder1=nn.Sequential(*encoder_block(n,nf))

        self.encoder2=nn.Sequential(*encoder_block(nf,nf*2))

        self.encoder3 =nn.Sequential(*encoder_block(nf*2, nf * 4))


        ##4 decoder
        self.decoder1 =nn.Sequential(*decoder_block(nf*4,nf*2))
        self.decoder2 =nn.Sequential(*decoder_block(nf * 4, nf * 2))
        self.decoder3 = nn.Sequential(*decoder_block(nf * 3, nf ))
        self.decoder4 =nn.Sequential(*decoder_block((nf+n), n))

        self.last = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=4, stride=2, padding=1)

    def forward(self,x):
        #2*8*32->2*32*64
        x_1=self.ConvT(x)
        x_2=self.Conv1(x_1)
        x_3=self.LeakyRELU(x_2)

        #3 encoder
        x1=self.encoder1(x_3)
        x2=self.encoder2(x1)
        x3=self.encoder3(x2)

        #4 decoder
        y1=self.decoder1(x3)
        y2=self.decoder2(torch.cat((y1,x2),dim=1))
        y3=self.decoder3(torch.cat((y2,x1),dim=1))
        y4=self.decoder4(torch.cat((y3,x_3),dim=1))

        #conv2+tanh
        y=self.last(y4)
        # y=nn.Tanh(y)
        return y

generator=Generator()
print(generator)

#save model
torch.save(generator,'./models/generator.pth')
print('模型保存成功!')




