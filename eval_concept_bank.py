import argparse
from sklearn.metrics import confusion_matrix
from torch import nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from backbones.meta_clip import MetaCLIP
from custom_datasets.broden_dataset import BrodenDataset
from concept_bank.concept_bank import ConceptBank
import datetime
import pickle


def parse_args(args = None):
    parser = argparse.ArgumentParser(description='Create populated concept banks')
    parser.add_argument('--name', type=str,default='main', help="""Base Name for the concept bank""")
    parser.add_argument('--model', default='MetaCLIP', type=str, help="""type of backbone model. Choose from [MetaCLIP | FLAVA]""")
    parser.add_argument('--concepts', default='Broden', type=str, help="""Set of concepts to use. Choose from [Broden]""")
    parser.add_argument('--device', default='auto', type=str, help="""Device""")
    parser.add_argument('--seed', type=int,default=123, help="""Random seed.""")

    if args == None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)     

    return args

def prepare(name = "main", model = "MetaCLIP", concept_set = "Broden", device = "auto", seed = 123):

    print("Preparing Evaluating concept bank")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        device = "cpu"


    if model == "MetaCLIP":
        backbone = MetaCLIP(device = device)
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    concept_bank = ConceptBank(id=name)

    eval_concept_bank(concept_bank=concept_bank, model=backbone, concept_set=concept_set, device=device, seed=seed)

def eval_concept_bank(concept_bank, model, concept_set = "Broden", device = "auto", seed = 123):
    print("Evaluating concept bank")

    summary = {}

    for concept_space_name, concept_space in concept_bank.concept_spaces.items():

        targets = []
        predictions = []

        for idx, (concept_name, concept) in enumerate(concept_space.concepts.items()):
            print(f"##### CONCEPT: {concept_name} #####")
            if concept_set == "Broden":
                from custom_datasets.broden_dataset import BrodenDataset
                dataset = BrodenDataset(
                    root_dir="datasets/",
                    category=concept_space_name,
                    target=concept_name,
                    resolution=224,
                    n_samples=2
                )

            for image, label in dataset:

                if isinstance(model,MetaCLIP):
                    image_encodings = model.encode_pyramid_image(image)
                    concept_pred_mask = []

                    for image_encoding in image_encodings:
                        
                        image_shape = image.shape
                        N_patches = int(np.sqrt(image_encoding.shape[0]))

                        _concept_pred_mask = concept_space.forward(image_encoding) ### since we have 0 as background                
                        _concept_pred_mask = _concept_pred_mask.reshape(N_patches, N_patches, -1).permute(2,0,1).unsqueeze(0)
                        
                        N_classes = _concept_pred_mask.shape[1]

                        ### scale up concept mask to image size
                        _concept_pred_mask = nn.functional.interpolate(
                            _concept_pred_mask,
                            scale_factor=(image_shape[1]/N_patches, image_shape[2]/N_patches),
                            mode="bilinear",
                            align_corners=False,
                        )
                        concept_pred_mask.append(_concept_pred_mask)

                    concept_pred_mask = torch.stack(concept_pred_mask)
                    concept_pred_mask = concept_pred_mask.amax(dim=0)
                else:
                    raise NotImplementedError(f"Model {model} not implemented")

                #print(f"Concept pred mask: {concept_pred_mask.shape} min: {concept_pred_mask.amin(dim=(2,3))} max: {concept_pred_mask.amax(dim=(2,3))}", )

                target = ((label > 0.5).long() * (idx))
                target[label < 0.5] = -1 ## background is now -1

                prediction = concept_pred_mask#.argmax(dim=1)

                precision = (prediction.argmax(dim=1) == target).sum() / (target != -1).sum()

                print("Precision: ", precision)

                #print(f"Target: {target.shape} min: {target.amin()} max: {target.amax()} {idx}", )
                #print(f"Prediction: {prediction.shape} min: {prediction.amin()} max: {prediction.amax()}", )

                targets.append(target)
                predictions.append(prediction)

        print("### EVALUATION ###")

        targets = torch.cat(targets).squeeze()
        predictions = torch.cat(predictions).squeeze()

        targets = targets.reshape(-1)
        predictions = predictions.permute(1,0,2,3).reshape(predictions.shape[1], -1)

        non_background = targets != -1

        targets = targets[non_background]
        predictions = predictions[:, non_background]

        print(f"Targets: {targets.shape} min: {targets.amin()} max: {targets.amax()}", )
        print(f"Predictions: {predictions.shape} min: {predictions.amin()} max: {predictions.amax()}", )
        
        cf_matrix = confusion_matrix(targets, predictions.argmax(dim=0))

        print(f"Confusion Matrix: {cf_matrix}")

        summary[concept_space_name] = {
            "confusion_matrix": cf_matrix
        }

    store_summary(summary)
    return summary


def store_summary(summary):
    now = datetime.datetime.now()
    file_name = f"evals/concept_bank_eval_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"

    with open(file_name, 'wb') as f:
        pickle.dump(summary, f, pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    args = parse_args(None)
    prepare(**args)