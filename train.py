import torch as t
from torch import autograd
import os
from torch import utils
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib
matplotlib.use('AGG')
import  matplotlib.pyplot as plt
from torchvision import transforms
from datasets import dataset,Celeba
from models import Generator,Discriminator
from utils import weight_init
from gradcam import *
class Network(object):
    def __init__(self,opt):
        #model parameterdim
        self.z=opt.z
        self.outch=opt.out_channel
        #dataset parameter
        self.dataset=opt.dataset
        self.dataset_add=opt.dataset_add
        self.c_size=opt.c_size
        self.batch=opt.batch
        self.size=opt.size
        #train parameter
        self.lr_g=opt.lr_G
        self.lr_d=opt.lr_D
        self.beta1=opt.beta1
        self.beta2=opt.beta2
        self.epoch = opt.epoch
        self.check_epoch=opt.check_epoch
        self.final_epoch=opt.final_epoch
        self.iter=opt.iter
        self.n_critic=opt.n_critic
        self.iter_img=opt.iter_img
        self.iter_mod=opt.iter_mod
        #save parameter
        self.images=opt.images
        self.plots=opt.plots
        self.models=opt.models
        self.visual_img=opt.visual_img
        self.cuda=opt.cuda
        #转换为cuda模式
        self.device=t.device('cuda' if t.cuda.is_available() else 'cpu',self.cuda)

        #参数设置
        self.lambda_p=1

        #引入模型
        self.G = Generator(self.z, self.outch).to(self.device)
        self.D = Discriminator(self.outch).to(self.device)

        #中途训练判断
        if self.epoch!=1:
            self.G.load_state_dict(t.load(os.path.join(self.models,'generator_%d.pth'%self.epoch)))
            self.D.load_state_dict(t.load(os.path.join(self.models,'discriminator_%d.pth'%self.epoch)))
       # else:  #模型权重初始化
        #    self.G.apply(weight_init)
         #   self.D.apply(weight_init)
        #定义优化器
        self.optimizer_g=Adam(self.G.parameters(),lr=self.lr_g,betas=(self.beta1,self.beta2))
        self.optimizer_d=Adam(self.D.parameters(),lr=self.lr_d,betas=(self.beta1,self.beta2))

        #加载数据集
        self.transform=transforms.Compose([transforms.CenterCrop(self.c_size),
                                           transforms.Resize(self.size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        datasets=Celeba(self.dataset_add,self.transform)
        self.dataloader=DataLoader(datasets,batch_size=self.batch,shuffle=True,drop_last=True)

        # 均值损失函数
        self.loss_mean=nn.MSELoss(size_average=False).to(self.device)

    # 全局损失函数定义,使用hingle损失
    def adv_loss_whole(self,x,label):
        if label == 'true':  # 当为真图像时
            loss = t.clamp(1 - x, 0)  # 采用hingle损失
        else:  # 当为假图像时
            loss = t.clamp(1 + x, 0)
        loss=loss.sum() / x.size(0)  # 针对batch求均值
        return  loss.to(self.device)

    #局部损失函数，只针对生成器
#    def adv_loss_part(self,x,att):
 #       ind = t.min(x, 1)[1].cpu()  # 取每一个图像中判别最不真的块的>索引
  #      index = t.cat((t.arange(self.batch).long(), ind), 0).view(2, -1)
   #     index = index.numpy().tolist()
    #    layer = att[index]  # 想得到64*64的矩阵，但得到64*2*64*64
     #   max_5 = t.topk(layer, 5)[1]  # 不写维数表示取每一行的前五个最>大值，并排序，返回值与索引
      #  loss = t.clamp(1 - x.gather(1, max_5), 0)  # 采用hingle损失
       # loss = loss.sum() / x.size(0)  # 针对batch求均值
        #return loss.to(self.device)

    def adv_loss_part(self,x):
        mean = x.mean(-1, keepdim=True)
        std = ((x - mean) ** 2 / 16).sum(-1)
        s=std.mean()
       # loss=t.Tensor([0]).to(self.device)
        #取50%的值
        if s>0 and s<0.03:
            x=x.cpu()
            b=x.kthvalue(6,dim=-1,keepdim=True)[0]
           # print(b.type())
            for i in range(x.size(0)):
                for j in range(x.size(1)):
                    if x[i][j]>b[i]:
                        x[i][j]=0
        #取25%的值
        if s>0.03 and s<0.04:
            x=x.cpu()
            b=x.kthvalue(10,dim=-1,keepdim=True)[0]
            for i in range(x.size(0)):
                for j in range(x.size(1)):
                    if x[i][j]>b[i]:
                        x[i][j]=0
        #取12.5%的值
        if s>0.04 :
            x=x.cpu()
            b=x.kthvalue(12,dim=-1,keepdim=True)[0]
            for i in range(x.size(0)):
                for j in range(x.size(1)):
                    if x[i][j]>b[i]:
                        x[i][j]=0
        loss=(-x).sum()/x.size(0)
        return loss

    def loss_plot(self):
        x=range(len(self.loss_t)) #损失函数的个数
        plt.plot(x,self.loss_t,label='loss_t')
        plt.plot(x,self.loss_f,label='loss_f')
        plt.plot(x, self.loss_G, label='loss_G')
        plt.plot(x, self.loss_D, label='loss_D')
        plt.xlabel('iter')
        plt.ylabel('loss')

        plt.legend(loc=0)
        plt.grid(True)

        plt.savefig(os.path.join(self.plots,'loss_%d.png'%self.final_epoch))
        plt.close()



    def train(self):
        #保存每个epoch中的损失函数值到列表中用于画图
        self.loss_t=[]
        self.loss_f=[]
        self.loss_g = []
        self.loss_p = []
        self.loss_G=[]
        self.loss_D=[]
        self.x=[]
      #进行训练
        for i in range(self.epoch,self.final_epoch+1):
           # if i == self.check_epoch:
               # self.optimizer_g.param_groups[0]['lr'] =0.0003
               # print('learn rate change')
            for iter,(img,label) in enumerate(self.dataloader):
                #train discriminato
                self.optimizer_d.zero_grad()
                img= img.to(self.device)
                t_out=self.D(img)
#                print(t_out.size())
                input = t.randn((self.batch, self.z)).to(self.device)  # 均匀分布中采样
                f_img = self.G(input)
                f_out = self.D(f_img.detach())

                loss_t = self.adv_loss_whole(t_out, label='true')
                loss_f = self.adv_loss_whole(f_out, label='false')
                    #fake image diacriminate

                loss_D=loss_t+loss_f

                loss_D.backward()
                self.optimizer_d.step()

                #train generator
                if iter%self.n_critic==0:
                    self.optimizer_g.zero_grad()
                    input=t.randn((self.batch,self.z)).to(self.device)
                    f_img=self.G(input)
                    r_out = self.D(f_img)
#                    print(f_img.size())
                    mean =r_out.mean(-1, keepdim=True)
                    std = ((r_out - mean) ** 2 / 64).sum(-1)
                    s=std.mean()
                    me = t.mean(mean)
                    st = ((mean - me) ** 2 / 64).sum()
                    if st>0.07 :
                       
                        loss_G = self.adv_loss_whole(r_out, label='true')
                        loss_G_=loss_G
                    else :
                        
                        loss_G = self.adv_loss_part(r_out)
                        loss_G_=self.adv_loss_whole(r_out, label='true')

                    #loss_g = self.loss_mean(r_out, label).sum()/self.batch  #均值损失
                    loss_G.backward()
                    self.optimizer_g.step()

                #打印损失信息
                if iter %self.iter==0:
                    print('[Epoch %d/%d][Batch %d/%d] D_loss:%f,d_true:%f,d_false:%f,G_loss:%f,G_lossw:%f,s:%f,st:%f'%
                          (i,self.final_epoch,iter,len(self.dataloader),loss_D.item(),loss_t.item(),
                           loss_f.item(),loss_G.item(),loss_G_.item(),s,st))

                #输出图片到指定文件夹
            if i% self.iter_img==0:
                print(mean.view(8, 8))
                print(std)
                print(me)
                print(s)
                print(st)
                path = os.path.join(self.images,'images_%d.png'%i)
                save_image(f_img,path,normalize=True)

                #保存损失函数值
                self.loss_t.append(loss_t.item())
                self.loss_f.append(loss_f.item())
                self.loss_G.append(loss_G_.item())
                self.loss_D.append(loss_D.item())

            #保存模型
            if i %self.iter_mod==0:
                path_g = os.path.join(self.models, 'generator_%d.pth' % i)
                path_d = os.path.join(self.models, 'discriminator_%d.pth' % i)
                t.save(self.G.state_dict(),path_g)
                t.save(self.D.state_dict(),path_d)
            #损失函数图
            if i==self.final_epoch:
                self.loss_plot()

    #生成相关epoch的测试图像
    def test(self,epo):
        with t.no_grad():
            path_mg = os.path.join(self.models, 'generator_%d.pth' % epo)
            path_md = os.path.join(self.models, 'discriminator_%d.pth' % epo)
            self.G.load_state_dict(t.load(path_mg))  # 加载相关epoch模型
            self.D.load_state_dict(t.load(path_md))
            for i in range(2000):
                input = t.randn((5, self.z)).to(self.device)
                img = self.G(input)
                r_out=self.D(img)
                mean =r_out.mean(-1, keepdim=True)
                std = ((r_out - mean) ** 2 / 64).sum(-1)
                s=std.mean()
                me = t.mean(mean)
                st = ((mean - me) ** 2 / 64).sum()
               # print(mean.view(8, 8))
               # print(std)
               # print('me',me)
               # print('st_in',s)
               # print('st_ou',st)
                path='data/1'
               # os.makedirs(path,exist_ok=True)
               # save_image(img, os.path.join(path, '1.jpg'), normalize=True)
                save_image(img[0],os.path.join(path,str(i*5+1)+'.jpg'), normalize=True)  # 保存图像
                save_image(img[1],os.path.join(path,str(i*5+2)+'.jpg'), normalize=True)
                save_image(img[2],os.path.join(path,str(i*5+3)+'.jpg'), normalize=True)
                save_image(img[3],os.path.join(path,str(i*5+4)+'.jpg'), normalize=True)
                save_image(img[4],os.path.join(path,str(i*5+5)+'.jpg'), normalize=True)
                print('generated images',i*5+5)

    def visual(self,epo):
       
            path_m = os.path.join(self.models, 'generator_%d.pth' % epo)
            path_d=os.path.join(self.models, 'discriminator_%d.pth' % epo)
            self.G.load_state_dict(t.load(path_m))  # 加载相关epoch模型
            self.D.load_state_dict(t.load(path_d))
           
            # for i in range(10):
            #     input = t.randn((5, self.z)).to(self.device)
            #     img = self.G(input)
            #     value=self.D(img)
            #     value=value.view((5,1,8,8)) #变换为相应规模
            #     value=value.clamp(-1,1)  #将其值裁剪为-1,1间的值，以便生成黑白图像
            #     path='data/value'
            #     os.makedirs(path,exist_ok=True)
            #     save_image(img,os.path.join(path,str(i*2+1)+'.jpg'), normalize=True)  # 保存图像
            #    save_image(value,os.path.join(path,str(i*2+2)+'.jpg'), normalize=True)
            #     print('generated images',i*2+2)
            for i in range(1,101):
                grad_cam = GradCam(model=self.D,target_layer_names=["layer4"],use_cuda=self.cuda)
                img = cv2.imread(os.path.join('data/1/',str(i)+'.jpg'), 1)
                img = np.float32(cv2.resize(img, (128,128))) / 255
                input = preprocess_image(img)
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested index.
                target_index = None
                mask = grad_cam(input, target_index)
               # show_cam_on_image(img, mask,os.path.join(self.visual_img,'camo'+str(i)+'.jpg'))
                value=self.D(input.cuda())
                value=-value.view(8,8)
                cam = np.maximum(value.detach().cpu().numpy(), 0)
                cam = cv2.resize(cam, (128, 128)).reshape(128,128,1)
               # print(img.shape,cam.shape)
               # cam=np.concatenate((img,cam),axis=1)
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
               # cv2.imwrite(os.path.join(self.visual_img,'cam'+str(i)+'.jpg'),np.uint8(cam*256))
               # value=value.view((1,1,8,8)) #变换为相应规模
               # value=value.clamp(-1,1) 
               # print(mask.shape,value.size()) 
                show_cam_on_image(img,cam,os.path.join(self.visual_img,'cam'+str(i)+'.jpg'))

               # save_image(value,os.path.join(self.visual_img,'wb'+str(i)+'.jpg'), normalize=True)
            #  gb_model = GuidedBackpropReLUModel(model=self.D, use_cuda=self.cuda)
             # gb = gb_model(input, index=target_index)
             # utils.save_image(t.from_numpy(gb), os.path.join(self.visual_img,'gb'+str(i)+'.jpg'))

             # cam_mask = np.zeros(gb.shape)
             # for i in range(0, gb.shape[0]):
              #  cam_mask[i, :, :] = mask

             # cam_gb = np.multiply(cam_mask, gb)
             # utils.save_image(t.from_numpy(cam_gb), os.path.join(self.visual_img,'cam_gb'+str(i)+'.jpg'))
