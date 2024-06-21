from typing import List, Dict
import pandas as pd
import numpy as np
import bnlearn as bn

def create_hierarchical_edges(parents_list: List[str], child_node: str):
  list_of_edges = []
  for parent in parents_list:
    list_of_edges.append((parent, child_node))
  return list_of_edges


def create_bayesian_model_from_data(data_set: pd.DataFrame):
    model = bn.structure_learning.fit(data_set)
    model = bn.parameter_learning.fit(model, data_set)
    return model