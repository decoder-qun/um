import torch.nn as nn
import numpy as np
def weights_init(m):
    classname=m.__class__.__name__
    # print(classname)
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,0.1)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1.0,0.1)
        m.bias.data.fill_(0)

def inv_lr_scheduler(param_lr,optimizer,iter_num,
                     gamma=0.0001,power=0.75,init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr']=lr*param_lr[i]
        i+=1
    return optimizer