from models.stacked_hourglass import StackedHourgalss
import torch.utils.data.dataloader
from torch.autograd import Variable as Variable
from data.handle.mpii import MPII

import os
print(os.getcwd())


train_loador = torch.utils.data.DataLoader(
    MPII('train'),
    batch_size = 6,
    shuffle = True,
    num_workers = 2
)

for epoch in range(2):
    for i,(input,target,meta) in enumerate(train_loador):
        if i>1:
            break
        inputs=Variable(input).float()
        targets=Variable(target).float()
        print(inputs,targets)


