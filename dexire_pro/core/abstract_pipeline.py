from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable, Set


class AbstractDexireProPipeline(ABC):
    
    # Steps of the pipeline. 
    @abstractmethod
    def preprocess_data(self):
        pass
        
    @abstractmethod
    def transform_embeddings_into_cluster(self): 
        pass
    
    @abstractmethod
    def extract_rules_from_clusters(self):
        pass
    
    @abstractmethod
    def create_probabilistic_graphical_model(self):
        pass
    
    @abstractmethod
    def explain_model_predictions(self):
        pass