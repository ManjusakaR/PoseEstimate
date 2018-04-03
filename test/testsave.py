import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
from test.testcifar10 import Net

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


transform=transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainLoador=torch.utils.data.DataLoader(dataset=trainset,batch_size=4,shuffle=True,num_workers=2)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testLoador=torch.utils.data.DataLoader(dataset=testset,batch_size=4,shuffle=False,num_workers=2)
classes=('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net=Net()
net.load_state_dict(torch.load('params.pkl'))

dataiter=iter(testLoador)
images,labels=next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ',' '.join('%5s'%classes[labels[j]] for j in range(4)))
outputs=net(Variable(images))

_,predicted=torch.max(outputs.data,1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]for j in range(4)))


