from .abstract_pipeline import AbstractDexireProPipeline
import tensorflow as tf
from ..model_utils import (identify_data_types,
                           generate_dataset_from_pandas, 
                           replace_bad_characters)
from ..clustering_explanations import ClusterAnalysis
from dexire.dexire import DEXiRE
import pandas as pd 
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import bnlearn as bn

def create_hierarchical_edges(parents_list: List[str], child_node: str):
  list_of_edges = []
  for parent in parents_list:
    list_of_edges.append((parent, child_node))
  return list_of_edges

def generate_edges(causal_factors, effect):
  edges = []
  for cause in causal_factors:
    edges.append((cause, effect))
  return edges

def from_effect_to_cause(causal_factors, effect):
  edges = []
  for cause in causal_factors:
    edges.append((effect, cause))
  return edges

def create_bayesian_model_from_data(data_set: pd.DataFrame, wlist=None, edges=None):
    model = bn.structure_learning.fit(data_set, white_list=wlist, bw_list_method='nodes')
    if edges is not None:
      model['model'].add_edges_from(edges)
    model = bn.parameter_learning.fit(model, data_set)
    return model

def print_CPDS(causes, evidence_dict, model):
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

def discretize_data(data: pd.DataFrame,
                    continuos_variables: List[str],
                    nbins: int = 10,
                    invert_order=True) -> Tuple[pd.DataFrame, Dict, Dict]:
  local_data = data.copy()
  discretize_transformer = {}
  dict_posible_values = {}
  for fac in continuos_variables:
    # discretizate
    array_values = data[fac].to_numpy()
    min = np.amin(array_values)
    max = np.amax(array_values)
    bins = np.linspace(min, max, num=nbins)
    discretize_transformer[fac] = bins
    discretized_data = np.digitize(array_values, bins, right=True)
    local_data[fac] = discretized_data
    # search for values
    dict_posible_values[fac] = sorted(np.unique(local_data[fac]))
  # return Bayesian network
  return local_data, discretize_transformer, dict_posible_values

class DexireProPipeline(AbstractDexireProPipeline):
    
    def __init__(self, 
                 model_path, 
                 feature_names,
                 class_name) -> None:
        self.model_path = model_path  # The pre-trained TensorFlow model for predicting embeddings
        self.feature_names = feature_names
        self.class_name = class_name
        super().__init__()
    
    def preprocess_data(self, categorical_feat,
                        numeric_feat, 
                        train_data,
                        test_data):
        # Implement the preprocessing steps here
        self.preprocessor = ColumnTransformer(
            [
                ('onehot', OneHotEncoder(), categorical_feat),
                ('scaler', StandardScaler(), numeric_feat)
            ],
            remainder='drop'
            )
        print(train_data.shape)
        X = self.preprocessor.fit_transform(train_data)
        X_arr = X.toarray()
        print(X_arr.shape)
        X_te = self.preprocessor.transform(test_data)
        X_te_arr = X_te.toarray()
        print(X_te_arr.shape)
        feature_names = self.preprocessor.get_feature_names_out()
        features_name_list = feature_names.tolist()
        feat_n = [replace_bad_characters(t) for t in features_name_list]
        return feat_n, X_arr, X_te_arr, train_data, test_data
    
    def transform_embeddings_into_cluster(self, 
                                          X_embedding, 
                                          train_data, 
                                          test_data, 
                                          cluster_column='embeddings'):
        # Implement the embedding transformation and clustering steps here
        self.ca = ClusterAnalysis()
        self.ca.automatically_choose_cluster_numbers(X_embedding, max_clusters=15)
        train_data[f'cluster_{cluster_column}'] = train_data[cluster_column].apply(lambda x: self.ca.predict(x.reshape(1,-1)))
        test_data[f'cluster_{cluster_column}'] = test_data[cluster_column.apply(lambda x: self.ca.predict(x.reshape(1,-1)))]
        return train_data, test_data
    
    def extract_rules_from_clusters(self,
                                    train_data: pd.DataFrame,
                                    test_data: pd.DataFrame,
                                    X_train_xai: np.array,
                                    X_test_xai: np.array,
                                    embedding_cols=None,
                                    preprocess=True
                                    ):
        # Implement the rule extraction from cluster steps here
        try:
            # load model
            print(f"Loading model at path: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("loaded model..")
            model_inputs = self.model.inputs
            numeric_feat, categorical_feat, embed_feat = identify_data_types(model_inputs)
            numeric_feat, categorical_feat, embed_feat = identify_data_types(model_inputs)
            # preprocess and cluster 
            if preprocess:
                feat_n, x_train_data, x_test_data, _, _ = self.preprocess_data(categorical_feat, 
                                                                 numeric_feat, 
                                                                 train_data, 
                                                                 test_data)
            else:
                x_train_data, x_test_data = train_data, test_data
                feat_n = self.feature_names
            if embedding_cols is not None:
                for emb_col in embedding_cols:
                    X_embedding = train_data[emb_col].to_numpy()
                    train_clusters, test_cluster = self.transform_embeddings_into_cluster(X_embedding,
                                                                        train_data,
                                                                        test_data, 
                                                                         emb_col)
                    x_train_data=np.append(x_train_data, 
                                           np.array(train_clusters[f'cluster_{emb_col}'].tolist(), dtype=np.float32), axis=1)
                    x_test_data=np.append(x_test_data, 
                                          np.array(test_cluster[f'cluster_{emb_col}'].tolist(), dtype=np.float32), axis=1)
                    feat_n += f'cluster_{emb_col}'
            train_ds = generate_dataset_from_pandas(train_data,
                                        numeric_feat,
                                        feature_columns=numeric_feat+categorical_feat,
                                        embedding_columns=embed_feat,
                                        target_col="label")
            train_ds = train_ds.batch(128)
            test_ds = generate_dataset_from_pandas(test_data,
                                        numeric_feat,
                                        feature_columns=numeric_feat+categorical_feat,
                                        embedding_columns=embed_feat,
                                        target_col="label")
            test_ds = test_ds.batch(128)
            # Extract rules
            self.dextractor = DEXiRE(
                                model=self.model,
                                feature_names=self.feature_names,
                                class_names=self.class_name,
                                explain_features=x_train_data
                                )
            print(f'Extracting rules...')
            self.rule_set = self.dextractor.extract_rules_at_layer(train_ds, layer_idx=-2)
            # assess rules
            y_true = test_data['label'].to_numpy()
            self.ans_dict = self.rule_set.assess_rule_set(
            X= x_test_data,
            y_true=y_true)
            self.x_test_data = x_test_data
            self.x_train_data = x_train_data
            y_pred = self.model.predict(test_ds)
            y_pred_model = np.rint(y_pred)
            self.y_pred_rules = self.rule_set.predict_numpy_rules(x_test_data)
            self.ans_dict.update({'fidelity': accuracy_score(y_pred_model, self.y_pred_rules)})
            self.ans_dict.update({'rule_len': len(self.rule_set)})
            # save assessment
            print(f'Rule extraction finished successfully')
            print(f'metrics: {self.ans_dict}')
            print('-----------------------------------------------')
        except Exception as e:
            print(f'Error: {e} extracting rules')
    
    def create_probabilistic_graphical_model(self, selected_df:pd.DataFrame, 
                                  numeric_features: List[str] = None,
                                  categorical_features: List[str] = None):
        # Implement the creation of the probabilistic graphical model steps here
        y_train_pred = self.rule_set.predict(self.x_train_data)
        selected_df["y_pred"] = y_train_pred
        discretized_df, discretize_dict, possible_vals = discretize_data(data=selected_df,
                    continuos_variables=numeric_features)
        discretized_df['cluster'] = self.x_train_data[:, -1]
        factor_df = discretized_df.loc[:, categorical_features+numeric_features+['cluster', 'y_pred']]
        self.bn_model = create_bayesian_model_from_data(factor_df)
        return self.bn_model
    
    def explain_model_predictions(self, evidence_dict, 
                                  causes: List[str],
                                  target_col: str = 'y_pred') -> None:
        # Implement the explanation generation steps here
        if self.model is not None:
            #predict
            bn.inference.fit(self.bn_model, variables=[target_col], 
                             evidence=evidence_dict)
            print_CPDS(causes, evidence_dict, self.bn_model)