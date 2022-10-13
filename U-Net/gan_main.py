import time

import torch
import numpy as np
from torch import nn
from scipy import io as scio
import h5py
from torch.utils.data import dataset
from Generator import generator
from Discriminator import discriminator
from matplotlib import pyplot as plt
import torchvision.utils as vutils
import os
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter(log_dir='logs')


#the number of antenna of BS
M=64
#the number of antenna of user
K=32
#delay
t=8

datafile='data\Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat'
# datasets=scio.loadmat(datafile)
datasets=h5py.File(datafile)
'''
['input_da', 'input_da_test', 'output_da', 'output_da_test']
input_da:量化后的Y
output_da:真实的信道矩阵H
'''
# print(datasets.keys())
# print(len(datasets['input_da']))#2(real+imag)
# print(datasets['input_da'][0].shape)#8*64*1547
# print(datasets['input_da_test'][0].shape)#8*64*664
# print(len(datasets['output_da']))#2
# print(datasets['output_da'][0].shape)#32*64*1547

#the trainsets of Y:batch_szie*channel*size
Y_train=torch.stack((torch.from_numpy(datasets['input_da'][0].transpose((2,0,1))),torch.from_numpy(datasets['input_da'][1].transpose((2,0,1)))),dim=1)
# print(Y_train.shape):1547*2*8*64
Y_test=torch.stack((torch.from_numpy(datasets['input_da_test'][0].transpose((2,0,1))),torch.from_numpy(datasets['input_da_test'][1].transpose((2,0,1)))),dim=1)

H_train=torch.stack((torch.from_numpy(datasets['output_da'][0].transpose((2,0,1))),torch.from_numpy(datasets['output_da'][1].transpose((2,0,1)))),dim=1)
H_test=torch.stack((torch.from_numpy(datasets['output_da_test'][0].transpose((2,0,1))),torch.from_numpy(datasets['output_da_test'][1].transpose((2,0,1)))),dim=1)

trainsets=torch.utils.data.TensorDataset(Y_train,H_train)
testsets=torch.utils.data.TensorDataset(Y_test,H_test)

#loader
trainloader=torch.utils.data.DataLoader(dataset=trainsets,shuffle=True,batch_size=64)
testloader=torch.utils.data.DataLoader(dataset=testsets,shuffle=False,batch_size=32)

#####net
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

optimizer_G=torch.optim.RMSprop(params=generator.parameters(),lr=2e-4)
optimizer_D=torch.optim.RMSprop(params=discriminator.parameters(),lr=2e-5)
#loss
criteration=nn.BCELoss()


# #draw picture(Y,fake_H,real_H)
# def draw_picture(generator,input,real_imag,t=0):
#     fake_imag=generator(input)
#     display_list=[np.squeeze(input.cpu()[0,:,:,:]),np.squeeze(real_imag.cpu()[0,:,:,:]),np.squeeze(fake_imag.cpu()[0,:,:,:])]
#     title=['Input Y','Real H','Fake H']
#     for i in range(3):
#         plt.subplot(1, 3, i + 1)
#         plt.title(title[i])
#         plt.imshow(display_list[i])
#         plt.axis("off")
#     plt.savefig(os.path.join("generated_img", "img_" + str(t) + ".png"))


#train
real_label=1
fake_label=0
def train(input,target,device,generator,discriminator,epoch,optimizer_G,optimizer_D):
    # the mode of trainning
    generator.train()
    discriminator.train()

    #initialize the grad of net
    generator.zero_grad()
    discriminator.zero_grad()

    #trainning
    input=input.float()
    target=target.float()
    fake_target=generator(input)#生成器的输入是量化后的Y input
    real_output=discriminator(target).view(-1)#判别器输入是H target,输出的是概率
    fake_output=discriminator(fake_target.detach()).view(-1)
    # print(real_output.dtype)
    #################D
    #compute loss of D
    b_size=target.size(0)
    D_loss1=criteration(real_output,torch.full(size=(b_size,),fill_value=real_label,device=device).float())#让真的尽可能被预测为1
    # print(D_loss1)
    # print(type(D_loss1))
    D_loss1.backward()#反向传播
    D_real=real_output.mean().item()#对一个batch里面的预测概率求平均

    D_loss2=criteration(fake_output,torch.full(size=(b_size,),fill_value=fake_label,device=device).float())#让假的尽可能被预测为0
    D_loss2.backward()#反向传播
    D_fake=fake_output.mean().item()

    #判别器总的损失
    D_loss=D_loss1+D_loss2

    #update D
    optimizer_D.step()

    ##############G
    # 注意:cGAN在传统GAN生成器损失基础上,加上(target,fake_target)之间的MSE损失(条件GAN)
    fake_target=generator(input)
    fake_output=discriminator(fake_target).view(-1)

    # compute the loss of generator
    #GAN
    # G_loss=criteration(fake_output,torch.full(size=(b_size,),fill_value=real_label,device=device).float())
    #cGAN
    L1=criteration(fake_output,torch.full(size=(b_size,),fill_value=real_label,device=device).float())
    L2=torch.mean(torch.abs(fake_target-target))
    G_loss=L1+L2*100
    # print(L1,L1.shape)
    # G_loss=L1+L2     #生成器的目标就是让假的更像真的

    #backward
    G_loss.backward()

    #mean prob
    G_fake=fake_output.mean().item()

    #update G
    optimizer_G.step()

    return D_real,D_fake,G_fake,D_loss,G_loss


D_Loss=[]
G_Loss=[]
img_list=[]
epochs=300

#offline training+online testing
print('-------------Start training!-------------')
for epoch in range(1,epochs+1):

    print('--------The{}Epoch'.format(epoch))
    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for i ,(input,target) in enumerate(trainloader):
        input,target=input.to(device),target.to(device)
        D_real, D_fake, G_fake, D_loss, G_loss=train(input,target,device,generator,discriminator,epoch,optimizer_G,optimizer_D)
        if i % 5 ==0:
            print('[Epoch %d/%d] [Batch %d/%d] [%d/%d] [D_loss:%f] [G_loss:%f]'%(epoch,epochs,i,len(trainloader),D_real,D_fake,D_loss.item(),G_loss.item()))
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')
    # #每训练完一轮进行测试
    # for ii,(test_input,test_target)in enumerate(testloader):
    #     if ii % 100==0:
    #         test_input,test_target=test_input.to(device),test_target.to(device)
    #         test_input=test_input.float()
    #         test_target=test_target.float()
    #         draw_picture(generator, test_input, test_target, t=epoch)

    writer.add_scalar('D_loss',D_loss,epoch)
    writer.add_scalar('G_loss',G_loss,epoch)
    D_Loss.append(D_loss)
    G_Loss.append(G_loss)



print('--------------Finish training!------------')
writer.close()
# G_loss.cpu()
# D_loss.cpu()
#
# plt.figure(figsize=(10, 20))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(G_Loss, label="G")
# plt.plot(D_Loss, label="D")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()












