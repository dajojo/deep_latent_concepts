### Tool dataset
from torch.utils.data import Dataset
import glob
from torchvision import transforms
from PIL import Image
import os
import random
from typing import List, Union
import torch

class CUBDataset(Dataset):
    def __init__(self, root_dir,species_number=0, n_samples = 4, transforms=None,transform=None, exclude_images = None,random_seed = 123):

        self.root_dir = root_dir
        self.transforms = transforms
        self.transform = transform

        self.species_number = species_number

        self.random_seed = random_seed

        ### dataset structure in root:
        ### images
        ###     0.albatross
        ###     ...
        ### segmentations
        ###     0.albatross
        ###     ...

        self.species_folder = None

        for foldername in glob.glob(root_dir+"/images/*/", recursive = True):
            if int(foldername.split("/")[-2].split(".")[0]) == species_number:
                print(f"found species folder: {foldername}")
                self.species_folder = foldername.split("/")[-2]
                break

        assert self.species_folder != None, "Species Folder not found. Specify a correct number and root folder"
        

        ### find all images..
        self.images_list = []
        for filename in glob.glob(root_dir+"/images/"+self.species_folder+'/*.jpg'):
            self.images_list.append(os.path.basename(filename)[:-4])

        print(f"Found {len(self.images_list)} images")


        if exclude_images != None:
            ### exlude these images
            self.images_list = list(filter(lambda image_id: image_id not in exclude_images, self.images_list ))
            
            print(f"Filtered remaining {len(self.images_list)} images")


        if random_seed != None:
            random.seed(random_seed)

        random.shuffle(self.images_list)

        self.images_list = self.images_list[:n_samples]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        random.seed(self.random_seed + idx)
        torch.manual_seed(self.random_seed + idx)

        image_id = self.images_list[idx]

        mask = Image.open(f"{self.root_dir}/segmentations/{self.species_folder}/{image_id}.png")
        mask = (transforms.ToTensor()(mask).squeeze(0)) > 0.5

        print(f"Mask {(mask).shape}")

        if len(mask.shape) > 2:
            ### weird extra...
            mask = mask.amax(dim=0)

        image = Image.open(f"{self.root_dir}/images/{self.species_folder}/{image_id}.jpg")
        image = transforms.ToTensor()(image)

        print(f"Image {(image).shape}")


        if self.transforms != None:            
            comb = self.transforms(torch.cat([image,mask.unsqueeze(0)]))
            image = comb[:3]
            mask = comb[3]

        if self.transform != None:
            image = self.transform(image)

        return image, mask
    

    def plot(self,n_samples = 4):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(n_samples, 2, squeeze=False,figsize=(25, 25))

        for idx,(image,label) in enumerate(self):
            if idx >= n_samples:
                break
            ax[idx,0].imshow(image.permute(1,2,0))
            ax[idx,1].imshow(label)

        plt.show()