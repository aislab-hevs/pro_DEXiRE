import tensorflow as tf
from .model_utils import (identify_data_types,
                           generate_dataset_from_pandas, 
                           replace_bad_characters,
                           transform_model_output)

from .probabilistic_graphical_model_utils import (print_CPDS,
                                                 create_bayesian_model_from_data)

from .preprocessing_utils import discretize_data


from .clustering_explanations import ClusterAnalysis
from dexire.dexire import DEXiRE
import dexire as dex
import pandas as pd 
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


from typing import List, Dict, Tuple, Any, Optional, Union
import traceback
import pandas as pd
import numpy as np
import bnlearn as bn

class DexireProPipeline:
    def __init__(self, 
                 model: tf.keras.models.Model,  
                 feature_names: List[str],
                 target_names: List[str],
                 categorical_features: Optional[List[str]]=None, 
                 numerical_features: Optional[List[str]] = None,
                 embedding_columns: Optional[List[str]]=None, 
                 ) -> None:
        self.model = model  # The pre-trained TensorFlow model for predicting embeddings
        self.feature_names = feature_names
        self.class_name = target_names
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.embedding_columns = embedding_columns
        self.data = {}
        self.rule_set = None
        self.bn_learn = None
        

    def preprocess_data_for_rules(self,  
                        train_data: pd.DataFrame,
                        test_data: Optional[pd.DataFrame] = None,
                        scale_numeric: bool = False) -> Dict[str, Dict[str, Any]]:
        # Implement the preprocessing steps here
        self.data["rules"] = {"raw": {"train_df": train_data, "test_df":  test_data}} 
        transforms = [('onehot', OneHotEncoder(sparse_output=False), self.categorical_features)]
        if scale_numeric:
          transforms.append(('scaler', StandardScaler(), self.numerical_features))
        self.preprocessor_rules = ColumnTransformer(
            transforms,
            remainder='drop'
            )
        X = self.preprocessor.fit_transform(self.data['rules']['raw']['train_df'])
        if test_data is not None:
          X_te = self.preprocessor.transform(test_data)
        else:
          X_te = None
        feature_names = self.preprocessor.get_feature_names_out()
        features_name_list = feature_names.tolist()
        feat_n = [replace_bad_characters(t) for t in features_name_list]
        self.data['rules'] = {'processed': {'x_train': X, 'x_test': X_te, 'feature_name': feat_n}}
        return self.data['rules']['processed']
    
    def transform_embeddings_into_cluster(self, 
                                          X_embedding: np.array,  
                                          train_data: pd.DataFrame,  
                                          test_data: pd.DataFrame, 
                                          cluster_column:str= 'embeddings', 
                                          max_clusters=15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Implement the embedding transformation and clustering steps here
        self.ca = ClusterAnalysis()
        self.ca.automatically_choose_cluster_numbers(X_embedding, max_clusters=max_clusters)
        train_cluster = train_data[cluster_column].apply(lambda x: self.ca.predict(x.reshape(1,-1)))
        test_cluster = test_data[cluster_column.apply(lambda x: self.ca.predict(x.reshape(1,-1)))]
        return train_cluster, test_cluster
    
    def extract_rules(self,
                      tf_model_input: Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]], 
                      X_train_xai: np.array,
                      y_train_xai: np.array,
                      tf_model_test_data: Union[tf.data.Dataset, Tuple[tf.Tensor, tf.Tensor]] = None, 
                      X_test_xai: np.array = None,
                      y_test_xai: np.array = None,
                      layer_list: Optional[Union[List[int], int]] = None,
                      xai_feature_names: Optional[List[str]] = None,
                      mode: Optional[dex.core.dexire_abstract.Mode] = dex.core.dexire_abstract.Mode.CLASSIFICATION):
        try:
            # load model
            if self.model is None:
              raise ValueError("Model is not loaded.")
            # Extract rules
            if xai_feature_names is None:
              xai_feature_names = self.feature_names
            self.dextractor = DEXiRE(
                                model=self.model,
                                feature_names=xai_feature_names,
                                class_names=self.class_name,
                                explain_features=X_train_xai,
                                mode = mode,
                                )
            print(f'Extracting rules...')
            if layer_list is None or isinstance(layer_list, list):
              self.rule_set = self.dextractor.extract_rules(tf_model_input,
                                                            y=y_train_xai,
                                                            layer_idx=layer_list)
            else:
              self.rule_set = self.dextractor.extract_rules_at_layer(tf_model_input, 
                                                                     y=y_train_xai,
                                                                   layer_idx=layer_list)
            # assess rules
            if X_test_xai is not None and y_test_xai is not None:
              y_true = y_test_xai
              self.ans_dict = self.rule_set.assess_rule_set(
                X= X_test_xai, 
                y_true=y_true)
              if tf_model_test_data is not None:
                y_pred = self.model.predict(tf_model_test_data)
                y_pred_model = transform_model_output(y_pred)
                self.y_pred_rules = self.rule_set.predict_numpy_rules(X_test_xai)
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
                                  categorical_features: List[str] = None,
                                  embedding_features: List[str] = [],
                                  target_col='y_pred') -> Optional[Any]:
        # Implement the creation of the probabilistic graphical model steps here
        discretized_df, discretize_dict, possible_vals = discretize_data(data=selected_df,
                    continuos_variables=numeric_features)
        factor_df = discretized_df.loc[:, categorical_features+numeric_features+embedding_features+[target_col]]
        bn_model = create_bayesian_model_from_data(factor_df)
        self.bn_learn = {"discretized_df": discretized_df, 
                         "discretize_dict": discretize_dict, 
                         'bn_model': bn_model}
        return bn_model
    
    def explain_model_predictions(self, evidence_dict, 
                                  causes: List[str],
                                  target_col: str = 'y_pred') -> None:
        # Implement the explanation generation steps here
        if self.model is not None:
            #predict
            prediction = bn.inference.fit(self.bn_model, variables=[target_col], 
                             evidence=evidence_dict)
            print(f'Prediction: {prediction}')
            print_CPDS(causes, evidence_dict, self.bn_model)
            
    def full_pipeline(self, 
                      train_df: pd.DataFrame,
                      test_df: pd.DataFrame = None,   
                      dict_steps_parameters: Dict[str, Any] = None) -> None:
      try:
        if dict_steps_parameters is not None:
          if 'data_preprocessing' in dict_steps_parameters:
            print('Data preprocessing...')
            self.preprocess_data_for_rules(train_df, 
                                           test_df,
              **dict_steps_parameters['data_preprocessing'])
          else: 
            self.data["rules"] = {"raw": {"train_df": train_df, "test_df":  test_df}} 
          if 'embedding_transformation' in dict_steps_parameters:
            print('Embedding transformation...')
            train_cluster, test_cluster = self.transform_embeddings_into_cluster(**dict_steps_parameters['embedding_transformation'])
            self.data['embedding'] = {'train_cluster': train_cluster, 
                                      'test_cluster': test_cluster}
          if 'rule_extraction' in dict_steps_parameters:
            print('Rule extraction...')
            if 'processed' not in self.data['rules']:
              print('Not processing')
              self.data['rules'] = {'processed': {'x_train': dict_steps_parameters['rule_extraction']['X_train_xai'], 
                                                  'x_test': dict_steps_parameters['rule_extraction']['X_test_xai'], 
                                                  'feature_name': dict_steps_parameters['rule_extraction']['xai_feature_names']}}
            self.extract_rules(**dict_steps_parameters['rule_extraction'])
            print('rules extracted successfully')
            y_pred_rules = self.rule_set.predict_numpy_rules(self.data['rules']['processed']['x_train'])
            selected_df = train_df.copy()
            selected_df = selected_df[self.feature_names]
            selected_df['y_pred'] = y_pred_rules
            self.data['selected_df'] = selected_df
            print('Finished rule extraction step')
          if 'probabilistic_graphical_model' in dict_steps_parameters:
            print('Probabilistic graphical model...')
            self.create_probabilistic_graphical_model(selected_df=self.data['selected_df'],
                                                      **dict_steps_parameters['probabilistic_graphical_model'])
        print('Pipeline execution finished successfully')
      except Exception as e:
        print(f'Error: {e} in the full pipeline')
        print(traceback.format_exc())
      