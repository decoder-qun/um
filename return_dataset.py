from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import os
from augment import RandAugmentMC

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_classlist(image_list):
    '''
    :param image_list: for example, each line of txt is real/blackberry/real_034_000315.jpg 12
    :return: num_per_cls_list: a list of number of per class
    in addition, label_num_dict[label]: a dictionary of {name of the label:total image number of the label}
    '''
    label_num_dict={}
    num_per_cls_list=[]
    label_list=[]
    with open(image_list) as f:
        for index, x in enumerate(f.readlines()):
            label=x.split(' ')[0].split('/')[-2] #blackberry
            if label not in label_list:
                label_list.append(label)
                label_num_dict[label]=0
            else:
                label_num_dict[label]+=1
    for key in label_num_dict:
        num_per_cls_list.append(label_num_dict[key])
    return num_per_cls_list

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# data/multi/real/aircraft_carrier/real_001_000114.jpg

def make_dataset_fromlist(file_path):
    '''
    :param file_path:
    :return:
    image_index: filename eg.sketch/whale/sketch_337_000147.jpg
    label_list: class index eg.[0,1,1,2,3,3,4,5,...,125]
    '''
    with open(file_path) as f:
        image_index=[x.split(' ')[0] for x in f.readlines()]
    with open(file_path) as f:
        label_list=[]
        selected_list=[]
        for index,x in enumerate(f.readlines()):
            label=x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(index)
        image_index=np.array(image_index)
        label_list=np.array(label_list)
    image_index=image_index[selected_list]
    return image_index,label_list


class Imagelists_VISDA(object):
    def __init__(self,file_path,image_path="data/multi/",transform=None,transform2=None,test=False):
        image_index,label_list=make_dataset_fromlist(file_path) #return filename , label index
        # print(len(image_index))
        self.image_index=image_index
        self.label_list=label_list
        self.transform=transform
        self.transform2=transform2
        self.loader=pil_loader
        self.image_path=image_path
        self.test=test
        # print(file_path,self.test)#self.transform,self.target_transform)

    def __getitem__(self,index):
        path=os.path.join(self.image_path,self.image_index[index]) # using index, you can find image name eg.sketch/whale/sketch_337_000147.jpg
        target=self.label_list[index] # using index, you can find label number eg.34
        img=self.loader(path)
        if self.transform is not None:
            img1=self.transform(img)
        if self.transform2 is not None:
            img2=self.transform2(img)
            return img1,img2,target
        if not self.test:
            return img1,target
        else:
            return img1,target,self.image[index]

    def __len__(self):
        return len(self.image_index)




def return_dataset(args):
    '''
    :param args: use of args including net,dataset,source,target
    :return: 5 loaders and a dictionary
    '''
    if args.net == 'alexnet':
        crop_size = 227
        bs = 32
    else:
        crop_size = 224
        bs = 24

    if torch.cuda.is_available():
        nw = 2
    else:
        nw = 0
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=2, m=10),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    file_path='data/txt/%s' % args.dataset
    image_path='data/%s/' % args.dataset
    source_filepath=os.path.join(file_path,'labeled_source_images_'+args.source+'.txt')
    target_filepath=os.path.join(file_path,'labeled_target_images_'+args.target+'_%d.txt' % args.num)
    target_unlabeled_filepath = os.path.join(file_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % args.num)
    target_val_filepath=os.path.join(file_path,'validation_target_images_'+args.target+'_3.txt')
    num_per_cls_list = return_classlist(source_filepath)
    # print(num_per_cls_list)
    print("%d classes in this dataset" % len(num_per_cls_list))

    #70358,378,23826,378,23826
    source_labeled_dataset=Imagelists_VISDA(source_filepath,image_path,transform=data_transforms['train'])
    target_labeled_dataset=Imagelists_VISDA(target_filepath,image_path,transform=data_transforms['train'])
    target_unlabeled_dataset=Imagelists_VISDA(target_unlabeled_filepath,transform=data_transforms['val'],transform2=data_transforms['self'])
    target_val_labeled_dataset=Imagelists_VISDA(target_val_filepath,transform=data_transforms['val'])
    target_test_unlabeled_dataset=Imagelists_VISDA(target_unlabeled_filepath,transform=data_transforms['test'])
    # print(len(target_labeled_dataset),len(target_unlabeled_dataset),len(target_val_labeled_dataset))
    source_labeled_loader=torch.utils.data.DataLoader(source_labeled_dataset,batch_size=bs,num_workers=nw,drop_last=True,shuffle=True)
    target_labeled_loader=torch.utils.data.DataLoader(target_labeled_dataset,batch_size=bs,num_workers=nw,drop_last=True,shuffle=True)
    target_unlabeled_loader=torch.utils.data.DataLoader(target_unlabeled_dataset,batch_size=bs,num_workers=nw,drop_last=True,shuffle=True)
    target_val_labeled_loader=torch.utils.data.DataLoader(target_val_labeled_dataset,batch_size=bs,num_workers=nw,drop_last=True,shuffle=True)
    target_test_unlabeled_loader=torch.utils.data.DataLoader(target_test_unlabeled_dataset,batch_size=bs,num_workers=nw,drop_last=True,shuffle=True)

    return source_labeled_loader,target_labeled_loader,target_unlabeled_loader,target_val_labeled_loader,target_test_unlabeled_loader,num_per_cls_list
