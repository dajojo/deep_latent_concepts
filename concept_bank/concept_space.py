from concept_bank.concept import Concept
import torch
from typing import Dict
import matplotlib.pyplot as plt


class ConceptSpace(torch.nn.Module):
    def __init__(self,name:str) -> None:
        self.name = name
        self.concepts:Dict[str,Concept] = {}
        pass

    def add_concept(self,concept:Concept):
        self.concepts[concept.name] = concept


    def plot_concepts_similarity(self):
        concepts = list(self.concepts.values())
        names = [c.name for c in concepts]
        encodings = torch.stack([c.encoding for c in concepts])

        encodings=encodings/encodings.norm(dim=1)[:,None]

        similarities = encodings @ encodings.T
        plt.imshow(similarities.detach().numpy())
        plt.colorbar()
        plt.xticks(range(len(names)), names, rotation=90)
        plt.yticks(range(len(names)), names)
        plt.show()

    def plot_concepts_probability(self, x: torch.Tensor):
        scores = self.forward(x)  # B x C
        names = [c.name for c in self.concepts.values()]

        if len(scores.shape) == 1:
            scores = scores.unsqueeze(0)
 

        num_bars = scores.shape[0]
        num_classes = scores.shape[1]
        bar_width = 0.35
        index = torch.arange(num_bars)

        for i in range(num_classes):
            plt.bar(index + i * bar_width, scores[:, i].detach().numpy(), bar_width, label=names[i])

        plt.xlabel('B')
        plt.ylabel('Probability')
        plt.xticks(index + (num_classes - 1) * bar_width / 2, range(num_bars))
        plt.legend()
        plt.show()

    def forward(self,x:torch.Tensor):
        encodings = torch.stack([c.encoding for c in self.concepts.values()])

        scores = encodings @ x.T

        scores = torch.softmax(scores, dim=0)

        return scores.T

        

