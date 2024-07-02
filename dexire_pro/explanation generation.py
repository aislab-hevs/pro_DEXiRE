from collections import OrderedDict
from dexire.core.rule_set import RuleSet
from typing import Any
from pgmpy.models import BayesianNetwork
import numpy as np
from model_utils import get_intermediate_model

def would_like_recipe(user_data_dict,
                      recipe_data_dict,
                      context_data_dict,
                      full_model,
                      ruleset: RuleSet=None,
                      bayesian_model: BayesianNetwork = None,
                      probability_threshold: float = 0.0):
  # preict user embedding
  total_data_dict = {}
  total_data_dict.update(user_data_dict)
  total_data_dict.update(context_data_dict)
  total_data_dict.update(recipe_data_dict)
  # prepare the context input
  prediction = full_model(total_data_dict)
  prediction = prediction.numpy()
  integer_prediction = np.rint(prediction)
  # get intermediate model:
  intermediate_model = get_intermediate_model(full_model, -6)
  x_embedding = intermediate_model(total_data_dict)
  x_numpy = x_embedding.numpy()
  # query the rule set
  if ruleset is not None:
    ans, dp = ruleset.predict(x_numpy, return_decision_path=True)
    print(dp)
    rule_list = ruleset.rules
  graph_list = []
  for i in range(integer_prediction.shape[0]):
    if i == 1:
      text_animated(f"I think you would like this recipe with probability: {prediction[i]}")
    else:
      text_animated(f"I think you would not like this recipe with probability: {1 - prediction[i]}\n")
    if ruleset is not None:
      # rule evaluation
      rule_idx = dp[i]
      for j in rule_idx:
        activated_rule = rule_list[j]
        probability_active_rule = activated_rule.proba
        text_animated(f"Because: {activated_rule} with probability: {probability_active_rule} %")
    if bayesian_model is not None:
      user_dict = {k: user_data_dict[k].numpy().item() for k in user_data_dict.keys()}
      context_dict = {k: context_data_dict[k].numpy().item() for k in context_data_dict.keys()}
      print(f"user_dict: {user_dict}")
      context_user_dict = user_dict.update(context_dict)
      user_context_df = pd.DataFrame(context_user_dict)
      print(user_context_df)
      proba_explanation = make_explanation_from_bn(bn_model=bayesian_model,
                              factor_list=decision_factors,
                              factor_values=values_list,
                              discretize_transfomer=transformer,
                              current_data=user_context_df,
                              predicted_target=1.0 if i == 1.0 else 0.0,
                              target_list=[0.0, 1.0])
      od = OrderedDict(sorted(proba_explanation.items(), key=lambda x: x[1]['probability'], reverse=True))
      print('\n')
      text_animated(f"Which means that the following factors impact your decision with probability described bellow.\n")
      nodes_list = []
      labels_list = []
      target_node = f"Prediction = {'Like 'if i == 1 else 'Dislike'}"
      for clave, valor in od.items():
        if valor['probability'] < probability_threshold:
          break
        nodes_list.append(f"{clave} = {valor['value'][0]}")
        labels_list.append(f"Probability = {np.round(valor['probability'], 3)}")
        text_animated(f"The variable: {clave} is taken the value: {valor['value'][0]}, impacting your decision with probability: {np.round(valor['probability'], 3)}\n")
      graph = create_decision_graph(nodes_list, target_node, labels_list)
      graph_list.append(graph)
  return integer_prediction, prediction, graph_list