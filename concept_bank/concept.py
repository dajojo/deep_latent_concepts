from typing import Union
import torch


class Concept(torch.nn.Module):
    def __init__(self,name:str,encoding:Union[torch.Tensor,None]) -> None:
        super().__init__()
        self.name = name
        self.encoding = torch.nn.Parameter(encoding)

    def forward(self, x):
        return self.encoding @ x