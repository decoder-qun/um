import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from return_dataset import return_dataset
from model.basenet import VGGBase, AlexNetBase
from model.resnet_meta import resnet34, ResNet32, resnet32, Predictor_meta, Predictor_deep_meta
from model.resnet import resnet34, Predictor, Predictor_deep
from loss import adentropy
from utils import weights_init, inv_lr_scheduler, to_var, plot_acc_loss, plot_acc, plot_loss
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
parser = argparse.ArgumentParser(description='meta-uda')
# TODO
parser.add_argument('--finish', type=str, default='F', choices=['T', 'F'], help='source part has trained or not')
parser.add_argument('--net', default='alexnet', help='which network to use as backbone')
parser.add_argument('--traditional_method', type=str, default='nometa', choices=['meta', 'nometa'],
                    help='use meta or not')
# TODO
parser.add_argument('--resume', action='store_true', help='resume from checkpoint',
                    default='/gdrive/MyDrive/Colab Notebooks/um2/um/save_model/temp.log_real_to_sketch_step_7500.pth')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging')
parser.add_argument('--val_interval', type=int, default=300, metavar='N',
                    help='how many batches to wait before logging')
parser.add_argument('--meta_resume', action='store_true', help='meta resume from checkpoint',
                    default=True)
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT', help='learning rate multiplication')
# TODO
parser.add_argument('--train_steps', type=int, default=50000, help='how many steps in train stage')
parser.add_argument('--meta_train_steps', type=int, default=20000, help='how many steps in meta train stage')
parser.add_argument('--dataset', type=str, default='multi', choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('-source', default='real', help='source dataset')
parser.add_argument('--target', default='sketch', help='target dataset')

parser.add_argument('--num', type=int, default=3, help='number of labeled examples in the target')
parser.add_argument('--threshold', type=float, default=0.95, help='loss weight')
parser.add_argument('--beta', type=float, default=1.0, help='loss weight')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')

parser.add_argument('--save_check', action='store_true', default=True, help='save checkpoint or not')
parser.add_argument('--save_model_path', type=str, default='./save_model', help='dir to save model')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda used in entropy and adentropy')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvement before terminating')
parser.add_argument('--early', action='store_false', default=True, help='early stopping on validation or not')
parser.add_argument('--method', type=str, default='MME', choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization, S+T is training only on labeled examples')
parser.add_argument('--log_file', type=str, default='./temp.log', help='dir to save checkpoint')

args = parser.parse_args(args=[])
print('Dataset:%s\tSource:%s\tTarget:%s\tLabeled num perclass:%s\tNetwork:%s\t' % (
args.dataset, args.source, args.target, args.num, args.net))
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
t = time.localtime()
record_file = os.path.join(record_dir, '%s_net_%s_%s_to_%s_num_%d_%d_%d' % (
args.method, args.net, args.source, args.target, t.tm_mon, t.tm_mday, t.tm_hour))
# log_file_name = './logs/'+'/'+args.log_file
# ReDirectSTD(log_file_name, 'stdout', True)# File will be deleted if already existing.

source_labeled_loader, target_labeled_loader, target_unlabeled_loader, target_val_labeled_loader, target_test_unlabeled_loader, num_per_cls_list = return_dataset(
    args)
# print(num_per_cls_list)

# effective weights: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
beta = 0.9999
effective_num = 1.0 - np.power(beta, num_per_cls_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)  # 1-ß/1-ß^n
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(num_per_cls_list)  # normalization
per_cls_weights = torch.FloatTensor(per_cls_weights)


def G_F1_meta(args):
    if args.net == 'resnet32':
        G = resnet32()
        inc = 512
    else:
        raise ValueError("Model cannot be recognized.")
    # F1 = Predictor_meta(num_class=len(num_per_cls_list), inc=inc)  # temp=T=0.05

    if "resnet" in args.net:
        F1 = Predictor_deep(num_class=len(num_per_cls_list), inc=inc)
    else:
        F1 = Predictor(num_class=len(num_per_cls_list), inc=inc)  # temp=T=0.05
    return G, F1


def G_F1(args):
    if args.net == 'alexnet':
        G = AlexNetBase()
        inc = 4096
    elif args.net == 'resnet34':
        G = resnet34()
        inc = 512
    elif args.net == 'vgg':
        G = VGGBase()
        inc = 4096
    else:
        raise ValueError("Model cannot be recognized.")
    if "resnet" in args.net:
        F1 = Predictor_deep(num_class=len(num_per_cls_list), inc=inc)
    else:
        F1 = Predictor(num_class=len(num_per_cls_list), inc=inc)  # temp=T=0.05
    return G, F1


if args.traditional_method == 'meta':
    G, F1 = G_F1_meta(args)
elif args.traditional_method == 'nometa':
    G, F1 = G_F1(args)
lr = args.lr
weights_init(F1)
G.to(device)
F1.to(device)

params = []

for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

# torch.cuda.manual_seed(1)
source_labeled_image = torch.FloatTensor(1).to(device)
source_labeled_label = torch.LongTensor(1).to(device)
target_unlabeled_image = torch.FloatTensor(1).to(device)
target_unlabeled_image2 = torch.FloatTensor(1).to(device)
target_labeled_image = torch.FloatTensor(1).to(device)
target_labeled_label = torch.LongTensor(1).to(device)
val_image = torch.FloatTensor(1).to(device)
val_label = torch.LongTensor(1).to(device)
eps = torch.FloatTensor(1).to(device)
weights = torch.FloatTensor(1).to(device)

source_labeled_image = Variable(source_labeled_image)
source_labeled_label = Variable(source_labeled_label)
target_unlabeled_image = Variable(target_unlabeled_image)
target_unlabeled_image2 = Variable(target_unlabeled_image2)
target_labeled_image = Variable(target_labeled_image)
target_labeled_label = Variable(target_labeled_label)
val_image = Variable(val_image)
val_label = Variable(val_label)

acc_list = []
loss_comb_list = []
loss_s_list = []
loss_u_list = []
loss_t_list = []
loss_val_list = []
with open('/gdrive/MyDrive/Colab Notebooks/um2/um/record/multi/MME/MME_net_alexnet_real_to_sketch_num_12_27_9',
          'r') as f:
    for line in f.readlines():
        current = line.split(",")[2]
        val_acc = current.split(" ")[4].replace("\n", "")
        acc_list.append(val_acc)


def main():
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    G.train()
    F1.train()

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    start_step = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_step = checkpoint['step']
            print("start_step:", start_step)
            G.load_state_dict(checkpoint['state_dict_G'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            F1.load_state_dict(checkpoint['state_dict_F'])
            optimizer_f.load_state_dict(checkpoint['optimizer_f'])
            best_acc = checkpoint['best_acc']
            print(best_acc)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    param_lr_g = []
    param_lr_f = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    # train
    # preparation of iteration of dataloader
    source_labeled_iter = iter(source_labeled_loader)
    target_unlabeled_iter = iter(target_unlabeled_loader)
    source_labeled_len = len(source_labeled_loader)
    target_unlabeled_len = len(target_unlabeled_loader)

    # start_step=args.train_steps+1
    if args.finish == 'F':
        for step in range(start_step, args.train_steps):
            optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
            optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)
            lr = optimizer_f.param_groups[0]['lr']

            # source dataset & target dataset
            # if all the source data has been iterated
            # in one step, next() produces a batchsize of data.
            # 2199/32 about 68 steps,745/32 about 23 steps, so they are all oversampling
            if step % source_labeled_len == 0:
                source_labeled_iter = iter(source_labeled_loader)
            if step % target_unlabeled_len == 0:
                target_unlabeled_iter = iter(target_unlabeled_loader)
            source_labeled = next(source_labeled_iter)
            target_unlabeled = next(target_unlabeled_iter)

            # first resize the torch tensor, making the tensor the same shape as the image, then copy the image
            source_labeled_image.resize_(source_labeled[0].size()).copy_(source_labeled[0])
            target_unlabeled_image.resize_(target_unlabeled[0].size()).copy_(target_unlabeled[0])
            target_unlabeled_image2.resize_(target_unlabeled[1].size()).copy_(target_unlabeled[1])
            source_labeled_label.resize_(source_labeled[1].size()).copy_(source_labeled[1])
            ns = source_labeled_image.size(0)
            nu = target_unlabeled_image.size(0)
            # print("ns,nu: ",ns,nu)

            zero_grad_all()
            # ns
            # nu
            # nu

            # ----------------ns-----------------  ------nu-------------- --------nu-------------
            image = torch.cat((source_labeled_image, target_unlabeled_image, target_unlabeled_image2), 0)
            # [72, 3, 224, 224]
            label = source_labeled_label
            # [24]
            feature = G(image)  # 64, 4096
            # [72, 512]
            predict = F1(feature)  # 64, 126(32+16+16)

            loss_s = F.cross_entropy(predict[:ns], label, reduction='mean')
            pseudo_label = torch.softmax(predict[ns:ns + nu].detach(), dim=-1)  # 32, 4096
            max_probs, target_u = torch.max(pseudo_label, dim=-1)  # max on 32 numbers
            mask = max_probs.ge(args.threshold).float()  # a list of 0 or 1
            loss_u = F.cross_entropy(predict[ns + nu:], target_u, reduction='none')
            loss_u = (loss_u * mask).mean()
            loss_comb = loss_s + args.beta * loss_u
            loss_comb.backward()
            optimizer_g.step()
            optimizer_f.step()
            zero_grad_all()

            unlabeled_feature = G(target_unlabeled_image)
            loss_t = adentropy(F1, unlabeled_feature, args.lamda)
            loss_t.backward()
            optimizer_g.step()
            optimizer_f.step()

            loss_comb_list.append(loss_comb)
            loss_s_list.append(loss_s)
            loss_u_list.append(loss_u)
            loss_t_list.append(loss_t)

            if step % args.val_interval == 0 and step > 0:
                log_train = 'Ep: {} lr: {} loss_comb: {:.6f} loss_s: {:.6f} loss_u: {:.6f} loss_t: {:.6f}' \
                    .format(step, lr, loss_comb, loss_s, loss_u, -loss_t)
                print(log_train)
                with open("loss_record", 'a') as f:
                    f.write(log_train + '\n')
                loss_val, acc_val = test(target_val_labeled_loader)
                G.train()
                F1.train()

                acc_list.append(acc_val)
                loss_val_list.append(loss_val)

                if acc_val >= best_acc:
                    best_acc = acc_val
                    counter = 0
                else:
                    counter += 1
                if args.early:
                    if counter > args.patience:
                        break;
                print('Best val accuracy %f, Current val accuracy %f\n' % (best_acc, acc_val))

                print('record %s' % record_file)
                with open(record_file, 'a') as f:
                    f.write('step %d, best %f, current val accuracy %f\n' % (step, best_acc, acc_val))
                # plot_acc_loss(args.val_interval,args.net,loss_val_list,acc_list)
                plot_acc(args.val_interval, args.net, acc_list)
                # plot_loss(args.val_interval,args.net,loss_val_list)
                G.train()
                F1.train()

                if args.save_check:
                    if step % args.save_interval == 0 and step > 0:
                        print('=> saving model')
                        if not os.path.exists(args.save_model_path):
                            os.makedirs(args.save_model_path)
                        filename = os.path.join(args.save_model_path, "{}_{}_{}_to_{}_step_{}.pth".
                                                format(args.net, args.log_file, args.source, args.target, step))
                        state = {'step': step + 1,
                                 'state_dict_G': G.state_dict(), 'optimizer_g': optimizer_g.state_dict(),
                                 'state_dict_F': F1.state_dict(), 'optimizer_f': optimizer_f.state_dict(),
                                 'best_acc': best_acc}
                        torch.save(state, filename)

    # --------------------------------------------------------------------------------------------------
    #
    #                                             meta-train
    #
    # --------------------------------------------------------------------------------------------------
    target_labeled_iter = iter(target_labeled_loader)
    target_labeled_len = len(target_labeled_loader)

    start_meta_step = 0

    if args.finish == 'T':
        for step in range(start_meta_step, args.meta_train_steps):
            # --------------------------------------------------------------------------------------------------
            #   trained with  source & unlabeled,loss=mean( (eps+weights)*(loss_s + ß * loss_u) ), update model
            # --------------------------------------------------------------------------------------------------
            # preparations for the final adaptive model G
            optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step + start_step, init_lr=args.lr)
            optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step + start_step, init_lr=args.lr)
            lr = optimizer_f.param_groups[0]['lr']

            # start for the copied model
            # every step we new a model, including G & F1
            if args.net == 'alexnet':
                meta_G = AlexNetBase()
                inc = 4096
                meta_F1 = Predictor(num_class=len(num_per_cls_list), inc=inc)
            elif args.net == 'resnet34':
                meta_G = resnet34()
                inc = 512
                meta_F1 = Predictor_deep(num_class=len(num_per_cls_list), inc=inc)
            elif args.net == 'vgg':
                meta_G = VGGBase()
                inc = 4096
                meta_F1 = Predictor(num_class=len(num_per_cls_list), inc=inc)
            meta_G.to(device)
            meta_F1.to(device)
            # copy params
            meta_G.load_state_dict(G.state_dict())
            meta_F1.load_state_dict(F1.state_dict())
            meta_G.train()
            meta_F1.train()

            meta_params = []
            for key, value in dict(meta_G.named_parameters()).items():
                if value.requires_grad:
                    if 'classifier' not in key:
                        meta_params += [{'params': [value], 'lr': args.multi,
                                         'weight_decay': 0.0005}]
                    else:
                        print("classifier not in key!")
                        meta_params += [{'params': [value], 'lr': args.multi * 10,
                                         'weight_decay': 0.0005}]

            optimizer_g_meta = optim.SGD(meta_params, momentum=0.9, weight_decay=0.0005, nesterov=True)
            optimizer_f_meta = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005,
                                         nesterov=True)
            lr_meta = optimizer_f_meta.param_groups[0]['lr']

            meta_param_lr_g = []
            meta_param_lr_f = []
            for param_group in optimizer_g_meta.param_groups:
                meta_param_lr_g.append(param_group["lr"])
            for param_group in optimizer_f_meta.param_groups:
                meta_param_lr_f.append(param_group["lr"])

            if step % source_labeled_len == 0:
                source_labeled_iter = iter(source_labeled_loader)
            if step % target_unlabeled_len == 0:
                target_unlabeled_iter = iter(target_unlabeled_loader)
            if step % target_labeled_len == 0:
                target_labeled_iter = iter(target_labeled_iter)
            source_labeled = next(source_labeled_iter)
            target_unlabeled = next(target_unlabeled_iter)
            target_labeled = next(target_labeled_iter)

            source_labeled_image.resize_(source_labeled[0].size()).copy_(source_labeled[0])
            source_labeled_label.resize_(source_labeled[1].size()).copy_(source_labeled[1])
            target_unlabeled_image.resize_(target_unlabeled[0].size()).copy_(target_unlabeled[0])
            target_unlabeled_image2.resize_(target_unlabeled[1].size()).copy_(target_unlabeled[1])
            ns = source_labeled_image.size(0)
            nu = target_unlabeled_image.size(0)

            zero_grad_all()
            image = torch.cat((source_labeled_image, target_unlabeled_image, target_unlabeled_image2), 0)
            image = to_var(image, requires_grad=False)
            label = source_labeled_label
            label = to_var(label, requires_grad=False)
            feature = meta_G(image)  # 64, 4096
            predict = meta_F1(feature)  # 64, 126

            y = torch.eye(len(per_cls_weights))  # 126*126
            label_one_hot = y[label].float()  # 32*126
            weights = torch.tensor(per_cls_weights).float()  # 126
            # weights=torch.tensor(per_cls_weights.clone().detach()).float()
            weights = weights.unsqueeze(0)
            weights = weights.repeat(label_one_hot.shape[0], 1) * label_one_hot  # 32*126
            weights = weights.sum(1)  # 32
            eps = torch.zeros(weights.size())
            weights = to_var(weights)
            eps = to_var(eps)
            w = weights + eps  # 32*1

            loss_s = F.cross_entropy(predict[:ns], label, reduce=False)
            pseudo_label = torch.softmax(predict[ns:ns + nu].detach(), dim=-1)  # 32, 126
            max_probs, target_u = torch.max(pseudo_label, dim=-1)  # max on 32 numbers
            mask = max_probs.ge(args.threshold).float()  # a list of 0 or 1
            loss_u = (F.cross_entropy(predict[ns + nu:], target_u, reduction='none')) * mask
            loss_comb = w * (loss_s + args.beta * loss_u)
            loss_comb_mean = torch.mean(loss_comb)
            meta_G.zero_grad()

            loss_comb_mean.backward()
            optimizer_g_meta.step()
            optimizer_f_meta.step()
            optimizer_g_meta.zero_grad()
            optimizer_f_meta.zero_grad()
            # grads = eps.grad.data
            # print(grads)
            # grads = torch.autograd.grad(loss_comb_sum, (meta_G.params()), only_inputs=True, create_graph=True, allow_unused=True)
            # grads_= ()
            # for i in range(len(grads)):
            #     if(grads[i]==None):
            #         grads_+=(grads[i],)
            #         print(i)

            # MME
            unlabeled_feature = meta_G(target_unlabeled_image)
            loss_t = adentropy(meta_F1, unlabeled_feature, args.lamda)
            loss_t.backward()
            optimizer_g_meta.step()
            optimizer_f_meta.step()
            optimizer_g_meta.zero_grad()
            optimizer_f_meta.zero_grad()

            # --------------------------------------------------------------------------------------------------
            #              trained with  target_labeled, update eps'<=loss_target
            # --------------------------------------------------------------------------------------------------
            target_labeled_image.resize_(target_labeled[0].size()).copy_(target_labeled[0])
            target_labeled_label.resize_(target_labeled[1].size()).copy_(target_labeled[1])
            image = target_labeled_image
            label = target_labeled_label
            image = to_var(image, requires_grad=False)
            label = to_var(label, requires_grad=False)
            feature = meta_G(image)  # 32, 4096
            predict = meta_F1(feature)  # 32, 126

            loss_target = F.cross_entropy(predict, label)  # , reduction='none')
            loss_target.backward()
            grad_eps = eps.grad.data
            new_eps = eps - 0.01 * grad_eps
            w = weights + new_eps
            del grad_eps  # , grads

            # --------------------------------------------------------------------------------------------------
            #              trained with  target_labeled, loss=mean( (eps'+weights)*loss_target ), update model
            # --------------------------------------------------------------------------------------------------
            G.train()
            F1.train()
            feature = G(image)  # 32, 4096
            predict = F1(feature)  # 32, 126
            loss_target_new = F.cross_entropy(predict, label, reduction='none')
            loss_target_new = (loss_target_new * w).mean()
            loss_target_new.backward()
            optimizer_g.step()
            optimizer_f.step()
            zero_grad_all()

            log_train = 'Ep: {} lr: {} lr_meta:{} loss_comb: {:.6f} loss_s: {:.6f} loss_u: {:.6f} loss_t: {:.6f} loss_target: {:.6f} loss_target_new: {:.6f}' \
                .format(step, lr, lr_meta, loss_comb_mean, loss_s.mean(), loss_u.mean(), -loss_t, loss_target,
                        loss_target_new)
            print(log_train)

            best_acc = 0.0
            if step % args.save_interval == 0 and step > 0:
                loss_val, acc_val = test(target_val_labeled_loader)
                G.train()
                F1.train()
                if acc_val >= best_acc:
                    best_acc = acc_val
                    counter = 0
                else:
                    counter += 1
                if args.early:
                    if counter > args.patience:
                        break;
                print('Best val accuracy %f, Current val accuracy %f\n' % (best_acc, acc_val))

                print('record %s' % record_file)
                with open(record_file, 'a') as f:
                    f.write('meta step %d, best %f, current val accuracy %f\n' % (step, best_acc, acc_val))

            G.train()
            F1.train()

            if args.save_check:
                if step % args.save_interval == 0 and step > 0:
                    print('=> saving meta model')
                    if not os.path.exists(args.save_model_path):
                        os.makedirs(args.save_model_path)
                    filename = os.path.join(args.save_model_path, "{}_{}_{}_to_{}_step_{}.pth".
                                            format(args.net, args.log_file, args.source, args.target, step))
                    state = {'step': step + 1,
                             'state_dict_G': G.state_dict(), 'optimizer_g': optimizer_g.state_dict(),
                             'state_dict_F': F1.state_dict(), 'optimizer_f': optimizer_f.state_dict()}
                    torch.save(state, filename)


def test(loader):
    G.eval()
    F1.eval()
    correct = 0
    size = 0
    num_class = len(num_per_cls_list)
    # output_all=np.zeros((0,num_class))
    confusion_matrix = torch.zeros(num_class, num_class)

    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            val_image.resize_(data[0].size()).copy_(data[0])
            val_label.resize_(data[1].size()).copy_(data[1])
            feature = G(val_image)
            predict = F1(feature)  # 32,126
            # output_all=np.r_[output_all,predict.numpy()]
            size += val_image.size(0)
            # pred1 = predict.max(1)[1]
            pred = torch.max(predict, dim=-1)[1]
            # print(pred)
            for t, p in zip(val_label.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred.eq(val_label).sum()
            test_loss = F.cross_entropy(predict, val_label, reduction='mean')
    print('Validation Set:Average loss: {:.4f}, Accuracy: {}/{}  F1: {:.4f}%'.format(test_loss, correct, size,
                                                                                     100. * float(correct) / float(
                                                                                         size)))
    return test_loss.data, 100. * float(correct) / size


if __name__ == '__main__':
    main()


