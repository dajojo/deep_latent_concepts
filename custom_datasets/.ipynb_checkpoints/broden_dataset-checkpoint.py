from torch.utils.data import Dataset
import glob
from torchvision import transforms
from PIL import Image
import os
import random
from typing import List, Union
import torch

class TextureDataset(Dataset):
    def __init__(self, root_dir,texture=None, n_samples = 4, transforms=None,transform=None, random_seed = 123):

        self.root_dir = root_dir
        self.transforms = transforms
        self.transform = transform

        self.texture = texture

        self.random_seed = random_seed

        ### find all images..
        self.images_list = []
        for filename in glob.glob(root_dir+"/images/dtd/*.jpg"):
            if texture in filename:
                self.images_list.append(os.path.basename(filename)[:-4])

        print(f"Found {len(self.images_list)} images")

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



        image = Image.open(f"{self.root_dir}/images/dtd/{image_id}.jpg")
        image = transforms.ToTensor()(image)

        print(f"Image {(image).shape}")

        mask = torch.ones((image.shape[1],image.shape[2]))

        print(f"Mask {(mask).shape}")


        if self.transforms != None:            
            comb = self.transforms(torch.cat([image,mask.unsqueeze(0)]))
            image = comb[:3]
            mask = comb[3]

        if self.transform != None:
            image = self.transform(image)

        return image, mask.unsqueeze(0)
    

    def plot(self,n_samples = 4):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(n_samples, 2, squeeze=False,figsize=(25, 25))

        for idx,(image,label) in enumerate(self):
            if idx >= n_samples:
                break
            print(image.shape)
            print(label.shape)
            ax[idx,0].imshow(image.permute(1,2,0))
            ax[idx,1].imshow(label.permute(1,2,0))

        plt.show()

#### TOOD color dataset

import os, errno, numpy, torch, csv, re, shutil, os, zipfile
from collections import OrderedDict
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from scipy import ndimage
from urllib.request import urlopen

class BrodenDataset(torch.utils.data.Dataset):
    '''
    A multicategory segmentation data set.

    Returns three streams:
    (1) The image (3, h, w).
    (2) The multicategory segmentation (labelcount, h, w).
    (3) A bincount of pixels in the segmentation (labelcount).

    Net dissect also assumes that the dataset object has three properties
    with human-readable labels:

    ds.labels = ['red', 'black', 'car', 'tree', 'grid', ...]
    ds.categories = ['color', 'part', 'object', 'texture']
    ds.label_category = [0, 0, 2, 2, 3, ...] # The category for each label
    '''
    def __init__(self, root_dir='datasets/broden', resolution=384,
            split='train', category=None,target=None,
            transform=None, transforms=None,
            download=False, size=None,
            broden_version=1, max_segment_depth=6,n_samples=16,seed=None):
        

        #assert category == None or negative_category == None , "Category or Negative category must be None"
        assert resolution in [224, 227, 384]

        ### Our use case only includes a single category.... or negative category: everything except the specified
        categories = [category]
        negative_categories = None #[negative_category]

        if seed != None:
            random.seed(seed)
        
        if download:
            ensure_broden_downloaded(root_dir, resolution, broden_version)
        self.directory = root_dir
        self.resolution = resolution
        self.resdir = os.path.join(root_dir, 'broden%d_%d' %
                (broden_version, resolution))
        self.loader = default_loader
        
        self.transform = transform
        self.transforms = transforms
        
        
        #self.include_bincount = include_bincount
        # The maximum number of multilabel layers that coexist at an image.
        self.max_segment_depth = max_segment_depth
        with open(os.path.join(self.resdir, 'category.csv'),
                encoding='utf-8') as f:
            self.category_info = OrderedDict()
            for row in csv.DictReader(f):
                self.category_info[row['name']] = row
        
        
        if categories is not None:
            # Filter out unused categories
            categories = set([c for c in categories if c in self.category_info])
            for cat in list(self.category_info.keys()):
                if cat not in categories:
                    del self.category_info[cat]

        elif negative_categories is not None:
            # Filter out specified category... leave the rest
            negative_categories = set([c for c in negative_categories if c in self.category_info])
            for cat in list(self.category_info.keys()):
                if cat in negative_categories:
                    del self.category_info[cat]



        categories = list(self.category_info.keys())
        self.categories = categories

        # Filter out unneeded images.
        with open(os.path.join(self.resdir, 'index.csv'),
                encoding='utf-8') as f:
            all_images = [decode_index_dict(r) for r in csv.DictReader(f)]
        self.image = [row for row in all_images
            if index_has_any_data(row, categories) and row['split'] == split]
        
        #print(all_images[0])
        
        if size is not None:
            self.image = self.image[:size]


        with open(os.path.join(self.resdir, 'label.csv'),
                encoding='utf-8') as f:
            self.label_info = build_dense_label_array([
                decode_label_dict(r) for r in csv.DictReader(f)])
            self.labels = [l['name'] for l in self.label_info]





        # Build dense remapping arrays for labels, so that you can
        # get dense ranges of labels for each category.
        self.category_map = {}
        self.category_unmap = {}
        self.category_label = {}
        self.target_code = None
        for cat in self.categories:
            with open(os.path.join(self.resdir, 'c_%s.csv' % cat),
                    encoding='utf-8') as f:
                c_data = [decode_label_dict(r) for r in csv.DictReader(f)]
            self.category_unmap[cat], self.category_map[cat] = (
                    build_numpy_category_map(c_data))
            self.category_label[cat] = build_dense_label_array(
                    c_data, key='code')
            print(f"category {cat} label: {self.category_label[cat]}")

            for label in self.category_label[cat]:
                if label["name"] == target:
                    self.target_code = label["number"]


        self.num_labels = len(self.labels)
        # Primary categories for each label is the category in which it
        # appears with the maximum coverage.
        self.label_category = numpy.zeros(self.num_labels, dtype=int)
        for i in range(self.num_labels):
            maxcoverage, self.label_category[i] = max(
               (self.category_label[cat][self.category_map[cat][i]]['coverage']
                    if i < len(self.category_map[cat])
                       and self.category_map[cat][i] else 0, ic)
                for ic, cat in enumerate(categories))


        print(f"target_code: {self.target_code}")

        random.shuffle(self.image)

        _image = []

        for record in self.image:
            mask = self.load_mask(record)

            if mask.mean() > 0.1:
                _image.append(record)

            if len(_image) >= n_samples:
                break


        self.image = _image#self.image[:n_samples]

    def __len__(self):
        return len(self.image)

    def load_mask(self,record):
        mask = numpy.zeros(shape=(
            record['sh'], record['sw']), dtype=int)
        
        mask = torch.zeros((record['sh'], record['sw']))

        depth = 0
        for cat in self.categories:
            for layer in record[cat]:
                    if isinstance(layer, int):             
                        if layer == self.target_code:
                            mask = mask + 1
                    else:
                        png = torch.Tensor(numpy.asarray(self.loader(os.path.join(
                            self.resdir, 'images', layer))))

                        #print(png[:,:,0].amax())
                        #print(png[:,:,0].amin())

                        _mask = (png[:,:,0] + png[:,:,1]).squeeze(-1)

                        #print(f"_mask: {_mask.shape} {self.target_code}")

                        mask[:,:] = (_mask == self.target_code).to(torch.int)# ((_mask - _mask.amin())/(_mask.amax() - _mask.amin())).to(torch.float)

                    depth += 1
        
        mask = torch.Tensor(mask)

        return mask

    def __getitem__(self, idx):
        record = self.image[idx]
        # example record: {
        #    'image': 'opensurfaces/25605.jpg', 'split': 'train',
        #    'ih': 384, 'iw': 384, 'sh': 192, 'sw': 192,
        #    'color': ['opensurfaces/25605_color.png'],
        #    'object': [], 'part': [],
        #    'material': ['opensurfaces/25605_material.png'],
        #    'scene': [], 'texture': []}


        image = self.loader(os.path.join(self.resdir, 'images',
            record['image']))

        image = transforms.ToTensor()(image)

        mask = self.load_mask(record)

        if self.transforms != None:            
            comb = self.transforms(torch.cat([image,mask.unsqueeze(0)]))
            image = comb[:3]
            mask = comb[3]

        if self.transform != None:
            image = self.transform(image)


        return (image, mask.unsqueeze(0))

    def plot(self,n_samples = 4):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(n_samples, 2, squeeze=False,figsize=(25, 25))

        for idx,(image,label) in enumerate(self):
            if idx >= n_samples:
                break
            #print(image.shape)
            #print(label.shape)
            ax[idx,0].imshow(image.permute(1,2,0))
            im = ax[idx,1].imshow(label.permute(1,2,0),vmin=0,vmax=1)

            plt.colorbar(im, ax=ax[idx,1])

        plt.show()

    

def build_dense_label_array(label_data, key='number', allow_none=False):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    result = [None] * (max([d[key] for d in label_data]) + 1)
    for d in label_data:
        result[d[key]] = d
    # Fill in none
    if not allow_none:
        example = label_data[0]
        def make_empty(k):
            return dict((c, k if c is key else type(v)())
                    for c, v in example.items())
        for i, d in enumerate(result):
            if d is None:
                result[i] = dict(make_empty(i))
    return result

def build_numpy_category_map(map_data, key1='code', key2='number'):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    results = list(numpy.zeros((max([d[key] for d in map_data]) + 1),
            dtype=numpy.int16) for key in (key1, key2))
    for d in map_data:
        results[0][d[key1]] = d[key2]
        results[1][d[key2]] = d[key1]
    return results

def index_has_any_data(row, categories):
    for c in categories:
        for data in row[c]:
            if data: return True
    return False

def decode_label_dict(row):
    result = {}
    for key, val in row.items():
        if key == 'category':
            result[key] = dict((c, int(n))
                for c, n in [re.match('^([^(]*)\(([^)]*)\)$', f).groups()
                    for f in val.split(';')])
        elif key == 'name':
            result[key] = val
        elif key == 'syns':
            result[key] = val.split(';')
        elif re.match('^\d+$', val):
            result[key] = int(val)
        elif re.match('^\d+\.\d*$', val):
            result[key] = float(val)
        else:
            result[key] = val
    return result

def decode_index_dict(row):
    result = {}
    for key, val in row.items():
        if key in ['image', 'split']:
            result[key] = val
        elif key in ['sw', 'sh', 'iw', 'ih']:
            result[key] = int(val)
        else:
            item = [s for s in val.split(';') if s]
            for i, v in enumerate(item):
                if re.match('^\d+$', v):
                    item[i] = int(v)
            result[key] = item
    return result

class ScaleSegmentation:
    '''
    Utility for scaling segmentations, using nearest-neighbor zooming.
    '''
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
    def __call__(self, seg):
        ratio = (1, self.target_height / float(seg.shape[1]),
                self.target_width / float(seg.shape[2]))
        return ndimage.zoom(seg, ratio, order=0)

def scatter_batch(seg, num_labels, omit_zero=True, dtype=torch.uint8):
    '''
    Utility for scattering semgentations into a one-hot representation.
    '''
    result = torch.zeros(*((seg.shape[0], num_labels,) + seg.shape[2:]),
            dtype=dtype, device=seg.device)
    result.scatter_(1, seg, 1)
    if omit_zero:
        result[:,0] = 0
    return result

def ensure_broden_downloaded(directory, resolution, broden_version=1):
    assert resolution in [224, 227, 384]
    baseurl = 'http://netdissect.csail.mit.edu/data/'
    dirname = 'broden%d_%d' % (broden_version, resolution)
    if os.path.isfile(os.path.join(directory, dirname, 'index.csv')):
        return # Already downloaded
    zipfilename = 'broden1_%d.zip' % resolution
    download_dir = os.path.join(directory, 'download')
    os.makedirs(download_dir, exist_ok=True)
    full_zipfilename = os.path.join(download_dir, zipfilename)
    if not os.path.exists(full_zipfilename):
        url = '%s/%s' % (baseurl, zipfilename)
        print('Downloading %s' % url)
        data = urlopen(url)
        with open(full_zipfilename, 'wb') as f:
            f.write(data.read())
    print('Unzipping %s' % zipfilename)
    with zipfile.ZipFile(full_zipfilename, 'r') as zip_ref:
        zip_ref.extractall(directory)
    assert os.path.isfile(os.path.join(directory, dirname, 'index.csv'))

def test_broden_dataset():
    '''
    Testing code.
    '''
    bds = BrodenDataset('dataset/broden', resolution=384,
            transform=transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor()]),
            transforms=transforms.Compose([
                        ScaleSegmentation(224, 224)
                        ]),
            include_bincount=True)
    loader = torch.utils.data.DataLoader(bds, batch_size=100, num_workers=24)
    for i in range(1,20):
        print(bds.label[i]['name'],
                list(bds.category.keys())[bds.primary_category[i]])
    for i, (im, seg, bc) in enumerate(loader):
        print(i, im.shape, seg.shape, seg.max(), bc.shape)

if __name__ == '__main__':
    test_broden_dataset()