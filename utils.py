import torch.cuda
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os

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

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x=x.cuda()
    return Variable(x,requires_grad=requires_grad)


def inv_lr_scheduler(param_lr,optimizer,iter_num,
                     gamma=0.0001,power=0.75,init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr']=lr*param_lr[i]
        i+=1
    return optimizer


def plot_acc_loss(val_interval,net,loss, acc):

    acc_,=plt.plot([i*val_interval for i in range(1,1+len(acc))],acc,label="accuracy")
    loss_,=plt.plot([i*val_interval for i in range(1,1+len(loss))], loss, label="loss")
    plt.title('validation accuracy&loss')
    plt.xlabel('epoches')
    plt.ylabel('accuracy&loss')

    # Create a legend for the first line.
    first_legend = plt.legend(handles=[acc_], loc=1)

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)

    # Create another legend for the second line.
    plt.legend(handles=[loss_], loc=4)

    plt.draw()
    if not os.path.exists("save_pic/"):
        os.makedirs("save_pic/")
    plt.savefig('save_pic/%s_epoch%d.jpg'%(net,val_interval*len(loss)))
    plt.show()

def plot_acc(val_interval,net, acc):
    acc_,=plt.plot([i*val_interval for i in range(1,1+len(acc))],acc,label="accuracy")
    plt.title('validation accuracy')
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.draw()
    if not os.path.exists("save_pic/"):
        os.makedirs("save_pic/")
    plt.savefig('save_pic/%s_epoch%d_acc.jpg'%(net,val_interval*len(acc)))
    plt.show()

def plot_loss(val_interval,net,loss):
    loss_,=plt.plot([i*val_interval for i in range(1,1+len(loss))], loss, label="loss")
    plt.title('validation loss')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.draw()
    if not os.path.exists("save_pic/"):
        os.makedirs("save_pic/")
    plt.savefig('save_pic/%s_epoch%d_loss.jpg'%(net,val_interval*len(loss)))
    plt.show()