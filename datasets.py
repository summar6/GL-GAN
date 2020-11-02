
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import Dataset
import torch as t
import glob
from PIL import Image
os.makedirs('./datasets',exist_ok=True)


def dataset(size,dataset=None,c_size=None,root=None):
    # 定义mnist数据集
    if dataset=='mnist':
        # 转化函数
        transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 正则化
        # 将真实数据标签设为1
        def target(label):
            label = t.ones(label.size())
            return label
        data = datasets.MNIST(root='./datasets', train=True, transform=transform, target_transform=target,download=True)
    elif dataset=='celeba':
        transform=transforms.Compose([transforms.CenterCrop((c_size,c_size)),
                                      transforms.Resize((size,size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        def target(label):
            label = t.ones(label.size())
        data=ImageFolder(root,transform=transform,target_transform=target)
    else:
        pass
    return data

#定义celeba数据集
class Celeba(Dataset):
    def __init__(self,root,transform):
        super(Celeba,self).__init__()
        self.root=root
        self.transform=transform
        self.path=sorted(glob.glob('%s/*.jpg'%root))
    def __getitem__(self, index):
        file=self.path[index]
       # filename=file.split('./')[-1]
        img=self.transform(Image.open(file))
        label=t.ones(1)
        return img,label
    def __len__(self):
        return len(self.path)
