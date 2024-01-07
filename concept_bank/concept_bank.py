from typing import List,Dict
from concept_bank.concept_space import ConceptSpace
import pickle
import os
import torch
import io

# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         return super().find_class(module, name)
 

class ConceptBank:
    def __init__(self,id:str,root_dir:str = "./banks",create_if_not_exist = True) -> None:
        self.id = id
        self.root_dir = root_dir
        self.concept_spaces:Dict[str,ConceptSpace] = {}

        self.exists = False

        if os.path.exists(self.root_dir+"/"+ self.id+ ".pkl"):
            self.load()
            self.exists = True
        elif create_if_not_exist:
            self.store()
            self.exists = True
        pass

    def add_concept_space(self,concept_space:ConceptSpace):
        self.concept_spaces[concept_space.name] = concept_space

    def load(self):
        path = self.root_dir+"/"+ self.id + ".pkl"
        with open(path,'rb') as f:

            _self = pickle.load(f)
            #_self = CPU_Unpickler(f).load()
            #_self = pickle.load(f)
            print(f"Retrieved concept bank from: {path}")
            self.__dict__.update(_self.__dict__)
            f.close()

    def store(self):

        for space_name, concept_space in self.concept_spaces.items():
            for concept_name, concept in concept_space.concepts.items():
                if concept.encoding != None:
                    concept.encoding = concept.encoding.cpu()

        path = self.root_dir+"/"+ self.id + ".pkl"
        with open(self.root_dir+"/"+ self.id + ".pkl",'wb') as f:
            print(f"Stored concept bank to: {path}")
            pickle.dump(self,f)
            f.close()