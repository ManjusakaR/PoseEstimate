import torch.utils.data.dataloader
from models.stacked_hourglass import StackedHourgalss
from torch.autograd import Variable as Variable
from data.handle.mpii import MPII
from utils.utils import AverageMeter,Flip,ShuffleLR
from utils.eval import Accuracy

stack_num=2
residual_num=3
channel_num=256
output_num=16
lr=2.5e-4
alpha=0.99
epsilon=1e-8
weight_decay=0.0
momentum=0.0
threads_num=1
batch_size=2
epoches_num=1

train_loador=torch.utils.data.DataLoader(MPII('train'),batch_size=batch_size,shuffle=True,num_workers=threads_num)

model=StackedHourgalss(stack_num,residual_num,channel_num,output_num)
model=model.cuda()

criterion=torch.nn.MSELoss()
optimizer=torch.optim.RMSprop(
    model.parameters(),lr=lr,alpha=alpha,eps=epsilon,weight_decay=weight_decay,momentum=momentum
)
criterion=criterion.cuda()

for epoch in range(epoches_num):

    Loss,Acc=AverageMeter(),AverageMeter()

    for i,(input,target,meta) in enumerate(train_loador):
        inputs=Variable(input).float().cuda()
        targets=Variable(target).float().cuda()
        output=model(inputs)

        loss=criterion(output[0],targets)
        for j in range(1,stack_num):
            loss+=criterion(output[j],targets)

        Loss.update(loss.data[0], input.size(0))
        Acc.update(Accuracy((output[stack_num - 1].data).cpu().numpy(), (targets.data).cpu().numpy()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Loss:'+Loss.avg,'Acc:'+Acc.avg())

torch.save(model.state_dict(),'params.pkl')

