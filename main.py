import torch.utils.data.dataloader
from models.stacked_hourglass import StackedHourgalss
from torch.autograd import Variable as Variable
from data.handle.mpii import MPII
from utils.utils import AverageMeter,Flip,ShuffleLR
from utils.eval import Accuracy
import ref

train_loador=torch.utils.data.DataLoader(MPII('train'),batch_size=ref.batch_size,shuffle=True,num_workers=ref.threads_num)

model=StackedHourgalss(ref.stack_num,ref.residual_num,ref.channel_num,ref.output_num)
model=model.cuda()

criterion=torch.nn.MSELoss()
optimizer=torch.optim.RMSprop(
    model.parameters(),lr=ref.lr,alpha=ref.alpha,eps=ref.epsilon,weight_decay=ref.weight_decay,momentum=ref.momentum
)
criterion=criterion.cuda()

for epoch in range(ref.epoches_num):

    Loss,Acc=AverageMeter(),AverageMeter()

    for i,(input,target,meta) in enumerate(train_loador):
        inputs=Variable(input).float().cuda()
        targets=Variable(target).float().cuda()
        inputs=Variable(input).float()
        targets=Variable(target).float()
        output=model(inputs)

        loss=criterion(output[0],targets)
        for j in range(1,ref.stack_num):
            loss+=criterion(output[j],targets)

        Loss.update(loss.data[0], input.size(0))
        Acc.update(Accuracy((output[ref.stack_num - 1].data).cpu().numpy(), (targets.data).cpu().numpy()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Loss:'+Loss.avg,'Acc:'+Acc.avg())

torch.save(model.state_dict(),'params.pkl')

