import pandas as pd
import numpy as np

from typing import List, Tuple, Dict


def discretize_data(data: pd.DataFrame,
                    continuos_variables: List[str],
                    nbins: int = 10,
                    invert_order=True) -> Tuple[pd.DataFrame, Dict, Dict]:
  local_data = data.copy()
  discretize_transformer = {}
  dict_possible_values = {}
  for fac in continuos_variables:
    # discretize
    array_values = data[fac].to_numpy()
    min = np.amin(array_values)
    max = np.amax(array_values)
    bins = np.linspace(min, max, num=nbins)
    discretize_transformer[fac] = bins
    discretized_data = np.digitize(array_values, bins, right=True)
    local_data[fac] = discretized_data
    # search for values
    dict_possible_values[fac] = sorted(np.unique(local_data[fac]))
  # return Bayesian network
  return local_data, discretize_transformer, dict_possible_values