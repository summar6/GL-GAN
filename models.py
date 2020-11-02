import torch.nn as nn
import torch as t
from utils import SpectralNorm
#generator创建
class Generator(nn.Module):
    # inch为输入channel,outch为输出channel,size为生成图片size
    def __init__(self,inch,outch):
        super(Generator,self).__init__()
        #创建一个函数块用于调用
        def layer_batch(inchnnel,outchnnel):
            layer=nn.Sequential(SpectralNorm(nn.ConvTranspose2d(inchnnel,outchnnel,4,2,1)),
                                  nn.BatchNorm2d(outchnnel),
                                  nn.ReLU(inplace=True))
            return layer
        #layer1:100--128*8*4*4
        self.layer1=nn.Sequential(SpectralNorm(nn.ConvTranspose2d(inch,128*8,4,1,0)),
                                  nn.BatchNorm2d(128*8),
                                  nn.ReLU(inplace=True))
        inchannel=128*8
        self.layer2=layer_batch(inchannel,inchannel//2) #layer2:128*8*4*4--128*4*8*8
        inchannel=inchannel//2
        self.layer3=layer_batch(inchannel,inchannel//2) #layer3:128*4*8*8--128*2*16*16
        inchannel = inchannel // 2
        self.layer4 = layer_batch(inchannel, inchannel // 2) #128*2*16*16--128*32*32
        inchannel = inchannel // 2
        self.layer5 = layer_batch(inchannel, inchannel // 2)  # layer5:128*32*32-64*64*64
        inchannel = inchannel // 2
        self.layer6=nn.Sequential(SpectralNorm(nn.ConvTranspose2d(inchannel, outch, 4, 2, 1)),
                        nn.Tanh())                        # 64*64-64*3*128*128

        #self.layer5 = nn.Sequential(nn.ConvTranspose2d(inchannel,outch,4,2,3),
                                  #  nn.Tanh()) #8*16*16-1*28*28   #针对mnist训练
        #加入权重控制参数,服从0,1均匀分布
        self.a1=nn.Parameter(t.rand((128*8,1,1)))
        self.a2 = nn.Parameter(t.rand((128*4,1,1)))
        self.a3 = nn.Parameter(t.rand((128*2,1,1)))
        self.a4 = nn.Parameter(t.rand((128,1,1)))
        self.a5 = nn.Parameter(t.rand((64, 1, 1)))



    def forward(self,input):
        input=input.view(input.size(0),input.size(1),1,1) #batch*100--batch*100*1*1
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # out=self.layer1(input)*self.a1
        # out=self.layer2(out)*self.a2
        # out=self.layer3(out)*self.a3
        # out=self.layer4(out)*self.a4
        # out = self.layer5(out) * self.a5
        # out=self.layer6(out)
        return out

#创建patch-gan判别器
class Discriminator(nn.Module):
    def __init__(self,inch):
        super(Discriminator,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(inch,64,4,2,1), #3*128*128-64*64*64
                                  nn.LeakyReLU(0.02,inplace=True))
        self.layer2=nn.Sequential(nn.Conv2d(64,128,4,2,1), #64*64*64-128*32*32
                                  nn.LeakyReLU(0.02,inplace=True))
        self.layer3=nn.Sequential(nn.Conv2d(128,256,4,2,1), #128*32*32-256*16*16
                                  nn.LeakyReLU(0.02,inplace=True))
        self.layer4 = nn.Sequential(SpectralNorm(nn.Conv2d(256,512,4,2,1)),  # 256*16*16-512*8*8
                                    nn.LeakyReLU(0.02, inplace=True))
        self.layer5 = nn.Sequential(SpectralNorm(nn.Conv2d(512,512,4,2,1)),  # 256*16*16-512*8*8
                                    nn.LeakyReLU(0.02, inplace=True))

        self.layer6=nn.Sequential(SpectralNorm(nn.Conv2d(512,1,1,1,0))) #512*8*8-1*8*8


    def forward(self,input):
        out=self.layer1(input)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out = self.layer5(out)
        out=self.layer6(out)
        out=out.view(out.size(0),-1)  #batch*64
        return out








