
from torch import Tensor, nn
from typing import Union, List

class ConceptBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,image):
        raise NotImplementedError
    
    # def preprocess(self,image):
    #     raise NotImplementedError

    # def get_facets(self,block_idx,image,facets: Union[List[str],str] = ["k","o"]):
    #     raise NotImplementedError

    # def scale_to_facet(self,block_id, facet = "k") -> Tensor:
    #     raise NotImplementedError