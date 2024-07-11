from typing import List, Dict, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import bnlearn as bn

def generate_causal_edges(causal_factors: List[str], effect: str) -> List[Tuple[str, str]]:
  edges = []
  for cause in causal_factors:
    edges.append((cause, effect))
  return edges

def create_hierarchical_edges(parents_list: List[str], child_node: str) -> List[Tuple[str, str]]:
  list_of_edges = []
  for parent in parents_list:
    list_of_edges.append((parent, child_node))
  return list_of_edges

def from_effect_to_cause(causal_factors: List[str], effect: str) -> List[Tuple[str, str]]:
  edges = []
  for cause in causal_factors:
    edges.append((effect, cause))
  return edges

def create_bayesian_model_from_data(data_set: pd.DataFrame, wlist=None, edges=None) -> Any:
    model = bn.structure_learning.fit(data_set, white_list=wlist, bw_list_method='nodes')
    if edges is not None:
      model['model'].add_edges_from(edges)
    model = bn.parameter_learning.fit(model, data_set)
    return model
  
def define_model_structure(edges: List[Tuple[str, str]]) -> Any:
  model = bn.make_DAG(edges)
  return model

def fit_model_with_data(model: Any, 
                        data_set: pd.DataFrame, 
                        methodtype: str = 'bayes', 
                        scoretype: str = 'bdeu',
                        ) -> Any:
  model = bn.parameter_learning.fit(model, data_set,
                                    methodtype=methodtype,
                                    scoretype=scoretype)
  return model
  

def get_CPDS_from_evidence(causes: List[str], evidence_dict: Dict[str, Any], model:Any) -> Dict[str, Any]:
  cpds = {}
  for cause in causes:
    if cause in evidence_dict.keys() and cause in model['model']:
      tem_evident_dict = evidence_dict.copy()
      del tem_evident_dict[cause]
      cpd = bn.inference.fit(model, variables=[cause], evidence=tem_evident_dict)
      cpds[cause] = cpd
  return cpds

def generate_visualization_probabilities(cpds: Dict[str, Any], evidence_dict: Dict[str, Any]):
  #TODO: Implement visualization of probabilities using libraries like matplotlib or seaborn
  pass

def print_CPDS(causes: List[str],
               evidence_dict: Dict[str, Any],
               model: Any) -> None:
  nodes_to_exclude = []
  for cause in causes:
    if cause not in model['model'] and cause in evidence_dict.keys():
      del evidence_dict[cause]
    nodes_to_exclude.append(cause)
  print(f"Excluded nodes: {nodes_to_exclude}")
  for cause in evidence_dict.keys():
    print(f"Cause: {cause}:")
    if cause in evidence_dict.keys():
      tem_evident_dict = evidence_dict.copy()
      del tem_evident_dict[cause]
      cpd = bn.inference.fit(model, variables=[cause], evidence=tem_evident_dict)
      print(cpd)
    print("------------------------------------------")