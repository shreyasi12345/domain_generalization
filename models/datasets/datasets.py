import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import fourier_transform
import random
from torchvision.transforms.functional import to_pil_image

from __main__ import args

#to perform augumentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'pre-train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        transforms.RandomGrayscale(p=0.2),lambda x: np.asarray(x)
        ]),
    'post-train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# function is not used
def image_transform(image, augmenters):
    """Transform an image

    Arguments:
        image : input image
        augmenters : data_transforms

    Returns:
        image : transformed image

    """
    # image preprocessing
    image = augmenters(image)
    return image


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(self, images, labels, domain, pre_transform_train=None,post_transform_train=None, test=None):
        """
        Args:
            images (list of string): Paths of the images.
            labels (list of int): labels of the images.
            domain (int): domain label
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.images = images  #paths
        self.labels = labels
        self.domain = domain
        self.pre_transform_train=pre_transform_train
        self.post_transform_train=post_transform_train
        self.test=test
        #self.post_transform = fourier_transform.get_post_transform()
        self.alpha=1.0

        #self.flat_names = []
        #self.flat_labels = []
        #self.flat_domains = []
        #for i in range(len(images)):
        #   self.flat_names += images[i]
        #   self.flat_labels += labels[i]
        #   self.flat_domains += [i] * len(images[i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.images[idx]
        lbls = self.labels[idx]
        domain = self.domain

        imgs = Image.open(imgs).convert('RGB')

        if self.pre_transform_train==None and self.post_transform_train==None:
            if self.test:
                imgs = self.test(imgs)
            return {'image': imgs, 'label': lbls, 'domain': domain}
        
        
        img_o = self.pre_transform_train(imgs)
        img_s, label_s = self.sample_image()
        img_s2o, img_o2s = fourier_transform.colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)


        img_o = self.post_transform_train(ensure_pil_image(img_o))
        img_s = self.post_transform_train(ensure_pil_image(img_s))
        img_s2o = self.post_transform_train(ensure_pil_image(img_s2o))
        img_o2s = self.post_transform_train(ensure_pil_image(img_o2s))



        #img_o, img_s = self.post_transform_train(img_o), self.post_transform_train(img_s)
        
        #img_s2o, img_o2s = self.post_transform_train(img_s2o), self.post_transform_train(img_o2s)
        imgs = [img_o, img_s, img_s2o, img_o2s]
        lbls = [lbls, label_s, lbls, label_s]
        
        return {'image': imgs, 'label': lbls, 'domain': domain}

    def sample_image(self):    
        img_idx = random.randint(0, len(self.images)-1)
        imgn_ame_sampled = self.images[img_idx]
        img_sampled = Image.open(imgn_ame_sampled).convert('RGB')
        label_sampled = self.labels[img_idx]
        return self.pre_transform_train(img_sampled), label_sampled
    
def get_image_label(category_list, label_list, domain_path):
    image_list = []
    lbl_list = []
    for category, label in zip(category_list, label_list):
        image_name = sorted(os.listdir(os.path.join(domain_path, category)))
        image_list.append([os.path.join(domain_path, category, e) for e in image_name])
        lbl_list.append([label]*len(image_name))
    image_list = np.array([x for e in image_list for x in e])
    lbl_list = np.array([x for e in lbl_list for x in e])
    return image_list, lbl_list

#helper function
def ensure_pil_image(img):
    # Convert torch tensor to PIL Image if it's a tensor
    if isinstance(img, torch.Tensor):
        img = to_pil_image(img)
    # Convert NumPy array to PIL Image
    elif isinstance(img, np.ndarray):
        # Check the shape of the image and handle accordingly
        if img.ndim == 3:
            # If it has 3 dimensions, check if it's already in (H, W, C) format
            if img.shape[0] == 1 and img.shape[1] == 1:
                # Reshape from (1, 1, 224) to a valid image format, e.g., grayscale
                img = np.squeeze(img)  # Remove dimensions of size 1
            elif img.shape[2] == 224:
                # Handle the case where the shape might be incorrect for an image
                # Assuming you want to convert it into a 224x224 grayscale image
                img = img.reshape(224, 224)
        elif img.ndim == 1:
            # Handle flattened images (reshape to 2D if needed)
            img = img.reshape(int(np.sqrt(img.size)), int(np.sqrt(img.size)))  # Reshape into a square image
        
        # Convert to PIL Image after reshaping
        img = Image.fromarray(img)
    
    return img

class PACS(object):
    """PACS train/val data

    7 categories

    images are from: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    These images are in the folder ./input/pacs/PACS/kfold/, e.g., ./input/pacs/PACS/kfold/APR/dog/***.jpg.
    training and validation split txt files are from:
    https://github.com/DeLightCMU/RSC/tree/master/Domain_Generalization/data/correct_txt_lists
    Put these txt files in ./input/pacs/PACS/kfold/
    """

    def __init__(self):
        data_path = os.path.join(args.datadir, 'pacs', 'PACS', 'kfold')

        domain_dic = {'art': 0, 'cartoon': 1, 'photo': 2, 'sketch': 3}
        
        self.category_list = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
      
        label_list = range(len(self.category_list))

        image_art, lbl_art = get_image_label(self.category_list, label_list, os.path.join(data_path, 'art_painting')) #art=apr
        print(image_art)
        image_cartoon, lbl_cartoon = get_image_label(self.category_list, label_list, os.path.join(data_path, 'cartoon')) #cartoon=Photos
        image_photo, lbl_photo = get_image_label(self.category_list, label_list, os.path.join(data_path, 'photo')) #photo=multispectral
        image_sketch, lbl_sketch = get_image_label(self.category_list, label_list, os.path.join(data_path, 'sketch'))

        dataset_art = BaseDataset(image_art, lbl_art, domain_dic['art'], None,None,data_transforms['test'])
        dataset_cartoon = BaseDataset(image_cartoon, lbl_cartoon, domain_dic['cartoon'],None,None, data_transforms['test'])
        dataset_photo = BaseDataset(image_photo, lbl_photo, domain_dic['photo'],None,None, data_transforms['test'])
        dataset_sketch = BaseDataset(image_sketch, lbl_sketch, domain_dic['sketch'],None,None, data_transforms['test'])

        # datasets for each domain
        #self.datasets = {'APR': dataset_art, 'Photos': dataset_cartoon, 'Multispectral': dataset_photo, 'sketch': dataset_sketch}
        self.datasets = {'art': dataset_art, 'cartoon': dataset_cartoon, 'photo': dataset_photo, 'sketch': dataset_sketch}
        

        # number of categories
        self.num_class = len(self.category_list)

        train_images_art = pd.read_csv(os.path.join(data_path, 'art_painting_train_kfold.txt'), header=None, sep=' ')
        image_train_art = data_path + '/' + train_images_art[0].values
        lbl_train_art = train_images_art[1].values - 1
        val_images_art = pd.read_csv(os.path.join(data_path, 'art_painting_crossval_kfold.txt'), header=None, sep=' ')
        image_val_art = data_path + '/' + val_images_art[0].values
        lbl_val_art = val_images_art[1].values - 1

        train_images_cartoon = pd.read_csv(os.path.join(data_path, 'cartoon_train_kfold.txt'), header=None, sep=' ')
        image_train_cartoon = data_path + '/' + train_images_cartoon[0].values
        lbl_train_cartoon = train_images_cartoon[1].values - 1
        val_images_cartoon = pd.read_csv(os.path.join(data_path, 'cartoon_crossval_kfold.txt'), header=None, sep=' ')
        image_val_cartoon = data_path + '/' + val_images_cartoon[0].values
        lbl_val_cartoon = val_images_cartoon[1].values - 1

        train_images_photo = pd.read_csv(os.path.join(data_path, 'photo_train_kfold.txt'), header=None, sep=' ')
        image_train_photo = data_path + '/' + train_images_photo[0].values
        lbl_train_photo = train_images_photo[1].values - 1
        val_images_photo = pd.read_csv(os.path.join(data_path, 'photo_crossval_kfold.txt'), header=None, sep=' ')
        image_val_photo = data_path + '/' + val_images_photo[0].values
        lbl_val_photo = val_images_photo[1].values - 1

        train_images_sketch = pd.read_csv(os.path.join(data_path, 'sketch_train_kfold.txt'), header=None, sep=' ')
        image_train_sketch = data_path + '/' + train_images_sketch[0].values
        lbl_train_sketch = train_images_sketch[1].values - 1
        val_images_sketch = pd.read_csv(os.path.join(data_path, 'sketch_crossval_kfold.txt'), header=None, sep=' ')
        image_val_sketch = data_path + '/' + val_images_sketch[0].values
        lbl_val_sketch = val_images_sketch[1].values - 1

        dataset_train_art = BaseDataset(image_train_art, lbl_train_art, domain_dic['art'], data_transforms['pre-train'],data_transforms['post-train'])
        dataset_train_cartoon = BaseDataset(image_train_cartoon, lbl_train_cartoon, domain_dic['cartoon'], data_transforms['pre-train'],data_transforms['post-train'])
        dataset_train_photo = BaseDataset(image_train_photo, lbl_train_photo, domain_dic['photo'], data_transforms['pre-train'],data_transforms['post-train'])
        dataset_train_sketch = BaseDataset(image_train_sketch, lbl_train_sketch, domain_dic['sketch'], data_transforms['pre-train'],data_transforms['post-train'])

        dataset_val_art = BaseDataset(image_val_art, lbl_val_art, domain_dic['art'],None,None, data_transforms['test'])
        dataset_val_cartoon = BaseDataset(image_val_cartoon, lbl_val_cartoon, domain_dic['cartoon'],None,None, data_transforms['test'])
        dataset_val_photo = BaseDataset(image_val_photo, lbl_val_photo, domain_dic['photo'],None,None, data_transforms['test'])
        dataset_val_sketch = BaseDataset(image_val_sketch, lbl_val_sketch, domain_dic['sketch'],None,None, data_transforms['test'])

        # datasets for each domain
        
        self.datasets_kfold = {'art': {'train': dataset_train_art, 'val': dataset_val_art}, 'cartoon': {'train': dataset_train_cartoon, 'val': dataset_val_cartoon}, 'photo': {'train': dataset_train_photo, 'val': dataset_val_photo}, 'sketch': {'train': dataset_train_sketch, 'val': dataset_val_sketch}}

        # number of data samples
        self.num_sample = {'art': {'train': len(image_train_art), 'val': len(image_val_art)}, 'cartoon': {'train': len(image_train_cartoon), 'val': len(image_val_cartoon)}, 'photo': {'train': len(image_train_photo), 'val': len(image_val_photo)}, 'sketch': {'train': len(image_train_sketch), 'val': len(image_val_sketch)}}


def get_data(domain_id):
    """Return Domain object based on domain_id

    Arguments:
        domain_id (string): domain name.

    Returns:
        Domain instance or None

    """
    if domain_id == 'pacs':
        return PACS()
    return None
