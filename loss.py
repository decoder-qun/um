import torch.nn.functional as F
import torch

def adentropy(F1,feature,lamda,eta=1.0):
    predict=F1(feature,reverse=True,eta=eta)
    predict=F.softmax(predict)
    loss_adent=lamda*torch.mean(torch.sum(predict*(torch.log(predict+1e-5))))
    return loss_adent