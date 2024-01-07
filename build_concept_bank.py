import torch
import numpy as np
import argparse
from backbones.meta_clip import MetaCLIP
from concept_bank.concept_bank import ConceptBank
from concept_bank.concept_space import ConceptSpace
from concept_bank.concept import Concept


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

def build_concept_bank(name="main",model="MetaCLIP",concept_set="Broden",device="auto",seed=123):

    print(f"Building concept bank")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        device = "cpu"


    if model == "MetaCLIP":
        model = MetaCLIP(device = device)
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    
    torch.manual_seed(seed)
    np.random.seed(seed)


    if concept_set == "Broden":
        from custom_datasets.broden_dataset import broden_concept_spaces
        concept = broden_concept_spaces

    concept_bank = ConceptBank(id=name)

    for concept_space_name, concept_names in concept.items():        
        concept_space = ConceptSpace(concept_space_name)

        for concept_name in concept_names:

            if concept_set == "Broden":
                from custom_datasets.broden_dataset import BrodenConceptDataset
                dataset = BrodenConceptDataset(
                    category=concept_space_name,
                    target=concept_name,
                )

            text_encodings = []

            for concept_text,target in dataset:
                text_encoding = model.encode_text(concept_text).squeeze()
                text_encodings.append(text_encoding)

            text_encodings = torch.stack(text_encodings)
            text_encoding = text_encodings.mean(dim=0)
            print(f"Mean Standard Deviation between {text_encodings.shape[0]} text encodings: {text_encodings.std(dim=0).mean()} for {concept_name}", )

            concept = Concept(
                name=concept_name,
                encoding=text_encoding,
            )

            concept_space.add_concept(concept)

        concept_bank.add_concept_space(concept_space)
        concept_bank.store()



if __name__ == "__main__":
    args = parse_args(None)
    build_concept_bank(**args)