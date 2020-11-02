#2019,10,29,yingliu
#encoding:Utf-8

'''关于手写数字生成案例的模型
应用块自注意力机制和channel加权权重'''

import argparse
import os
import torch as t
from train import Network
#设置项目中的参数
def args():
    parse=argparse.ArgumentParser()

    #model parameter
    parse.add_argument('--z',type=int,default=100,help='the size of input noise')
    parse.add_argument('--out_channel',type=int,default=3,help='the channel of output')

    #dataset parameter
    parse.add_argument('--dataset',type=str,default='celeba',choices=['mnist','celeba'],help='the name of dataset')
    parse.add_argument('--dataset_add',type=str,default='../CLSGAN/stargan/CelebA/img_align_celeba',help='the address of celeba dataset')
    parse.add_argument('--c_size',type=int,default=178,help='the size of centercrop')
    parse.add_argument('--batch',type=int,default=64,help='the batch size of input data')
    parse.add_argument('--size',type=int,default=128,help='the size of resized or generated images')

    #train parameter
    parse.add_argument('--mode',type=str,default='train',help='the mode of operation')
    parse.add_argument('--lr_G',type=float,default=0.0004,help='the learning rate of generator')
    parse.add_argument('--lr_D',type=float,default=0.0001,help='the learning rate of discriminator')
    parse.add_argument('--beta1',type=float,default=0.5,help='adam: decay of first order momentum of gradient')
    parse.add_argument('--beta2',type=float,default=0.999,help='adam: decay of first order momentum of gradient')
    parse.add_argument('--iter',type=int,default=20,help='the print interval')
    parse.add_argument('--n_critic',type=int,default=1,help='the update G after n update D')
    parse.add_argument('--iter_img',type=int,default=1,help='the generated images interval')
    parse.add_argument('--iter_mod',type=int,default=1,help='the saved models interval')
    parse.add_argument('--epoch',type=int,default=1,help='the start epoch about train')
    parse.add_argument('--check_epoch',type=int,default=20,help='the check epoch about train')
    parse.add_argument('--final_epoch',type=int,default=30,help='the final epoch about train')

    #save parameter
    parse.add_argument('--images',type=str,default='./save/images',help='the folder name of saving generated images')
    parse.add_argument('--plots',type=str,default='./save/plots',help='the folder name of saving the loss figure')
    parse.add_argument('--models',type=str,default='./save/models',help='the folder name of saving models')
    parse.add_argument('--visual_img',type=str,default='./data/img/',help='the path of visual img')
    parse.add_argument('--cuda',type=int,default=1,help='if use gpu or not')
    opt=parse.parse_args()
    return opt

#主函数
def main():
    opt=args()    #引入参数

    #创建相关路径
    os.makedirs(opt.images,exist_ok=True)
    os.makedirs(opt.plots,exist_ok=True)
    os.makedirs(opt.models,exist_ok=True)
    os.makedirs(opt.visual_img,exist_ok=True)

    #训练模型
    net=Network(opt)
    if opt.mode=='train':
        net.train()
    elif opt.mode=='test':
        net.test(opt.check_epoch)
    else :
        net.visual(opt.check_epoch)


if __name__=='__main__':
    main()
