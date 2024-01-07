from PIL import Image
import torch
#from transformers import AutoProcessor, AutoModel
import open_clip
import matplotlib.pyplot as plt

from backbones.concept_backbone import ConceptBackbone
import torchvision
import torch.nn.functional as F


class MetaCLIP(ConceptBackbone):
    def __init__(self,variant='metaclip-b32-400m',device = "cpu"):
        super().__init__()
        self.device = device
        assert variant == 'metaclip-b32-400m' or variant == 'metaclip-h14-fullcc2.5b', f"Variant {variant} not supported"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_400m')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo

        print(self.model)


    def encode_image(self, image):
        image = torchvision.transforms.ToPILImage()(image)
        image = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            return image_features

    def encode_pyramid_image(self, image, level = 5):
        image = torchvision.transforms.ToPILImage()(image)
        image = self.preprocess(image)


        original_size = image.shape[-1]

        image_level_features = []
        for i in range(level):
            level_size = original_size // (2**i)
            image_patches = image.unfold(0, 3, 3).unfold(1, level_size, level_size).unfold(2, level_size, level_size).reshape(-1, 3, level_size, level_size)
            image_patches = F.interpolate(image_patches, size=(224, 224), mode='bicubic', align_corners=False)

            with torch.no_grad():
                image_features = self.model.encode_image(image_patches)
                image_level_features.append(image_features)

        return image_level_features


    def encode_text(self, text):
        text = open_clip.tokenize(text)

        with torch.no_grad():
            text_features = self.model.encode_text(text)
            return text_features

    def forward(self,image,prompt):
        image = torchvision.transforms.ToPILImage()(image)
    
        image = self.preprocess(image).unsqueeze(0)

        text = open_clip.tokenize(prompt)

        with torch.no_grad():
            image_features,image_tokens = self.model._encode_image(image)
            text_features = self.model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            return image_features