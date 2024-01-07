from PIL import Image
import torch
from transformers import FlavaProcessor, FlavaModel
from transformers import FlavaFeatureExtractor, BertTokenizer

import open_clip
import matplotlib.pyplot as plt

from backbones.concept_backbone import ConceptBackbone
import torchvision
import torch.nn.functional as F


class Flava(ConceptBackbone):
    def __init__(self,variant='flava-full',device = "cpu"):
        super().__init__()
        self.device = device

        assert variant == 'flava-full', f"Variant {variant} not supported"

        self.model = FlavaModel.from_pretrained("facebook/"+variant)
        self.processor = FlavaProcessor.from_pretrained("facebook/"+variant)
        self.tokenizer = BertTokenizer.from_pretrained("facebook/"+variant)
        self.fe = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")


        print(self.model)

    def embed_text(self, text):
        return self.tokenizer(["This is " + desc for desc in [text]], return_tensors="pt", padding=True, max_length=5)

    def embed_image(self, image):
        return self.fe(image, return_tensors="pt")


    def forward(self,image,prompt):

        image_input = self.embed_image(image)
        text_tokens = self.embed_text(prompt)

        with torch.no_grad():
            # We take the output embedding for the CLS token for both encoders
            image_features = self.model.get_image_features(**image_input)[:, 0].float()
            text_features = self.model.get_text_features(**text_tokens)[:, 0].float()

        return image_features, text_features

        # image = torchvision.transforms.ToPILImage()(image)

        # inputs = self.processor(
        #     text=[prompt], images=[image], return_tensors="pt", padding="max_length", max_length=5
        # )
        
        # outputs = self.model(**inputs)
        # image_embeddings = outputs.image_embeddings # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
        # text_embeddings = outputs.text_embeddings # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768
        # multimodal_embeddings = outputs.multimodal_embeddings # Batch size X (Number of image patches + Text Sequence Length + 3) X Hidden size => 2 X 275 x 768
        # # Multimodal embeddings can be used for multimodal tasks such as VQA

        # print(image_embeddings.shape)
        # print(text_embeddings.shape)
        # print(multimodal_embeddings.shape)

        # return image_embeddings[0], text_embeddings[0], multimodal_embeddings[0]