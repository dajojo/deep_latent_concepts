import os.path
from typing import Any, Callable, List, Optional, Tuple
import random
from torchvision import transforms

from PIL import Image

from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt

import torch

class CocoDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        #root: str,
        #annFile: str,
        category: int = None,
        n_samples = 4,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        random_seed = 123
    ) -> None:
        root = "/Volumes/Intenso/datasets/coco-2017/images"
        annFile = "/Volumes/Intenso/datasets/coco-2017/annotations/instances_train2017.json"
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.category = category
        if category != None:
            self.ids = list(sorted(self.coco.catToImgs[category]))
        else:
            self.ids = list(sorted(self.coco.imgs.keys()))

        if random_seed != None:
            random.seed(random_seed)
        random.shuffle(self.ids)

        if n_samples != None:
            self.ids = self.ids[:n_samples]
            
    def _download_image(self,id:int) -> Image.Image:
        self.coco.download(tarDir=self.root,imgIds=[id])
        return self._load_image(id)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]

        try:
            image = self._load_image(id)
        except OSError as e:
            print(f"{e} Downloading image...")
            image = self._download_image(id)

        target_anns = self._load_target(id)

        ### ToTensor..
        image = transforms.ToTensor()(image)
        target = torch.zeros(image.shape[1:]).unsqueeze(-1)

        for ann in target_anns:
            if ann["category_id"] == self.category:
                mask = torch.Tensor(self.coco.annToMask(ann))
                print(mask.shape)
                target = torch.cat([target,mask.unsqueeze(-1)],dim=-1)

        target = (target.amax(dim=-1) > 0.1).to(torch.int)

        print(f"target: {target.shape} {target.dtype}")
        
        #target = transforms.ToTensor()(target)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self) -> int:
        return len(self.ids)
    
    def plot(self,n_samples = 4):
        fig, ax = plt.subplots(n_samples, 2, squeeze=False,figsize=(25, 25))

        for idx,(image,label) in enumerate(self):
            if idx >= n_samples:
                break
            ax[idx,0].imshow(image.permute(1,2,0))
            ax[idx,1].imshow(label)

        plt.show()


