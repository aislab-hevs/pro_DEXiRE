import graphviz
from typing import List, Dict

def create_decision_graph(list_nodes: List, target_node, labels: List[str]=None):
  dot = graphviz.Digraph(comment="Personalized Bayesian Network")
  for idx, node in enumerate(list_nodes):
    if labels is None:
      dot.edge(node, target_node)
    else:
      dot.edge(node, target_node, label=labels[idx])
  return dot