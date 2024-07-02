import tensorflow as tf
import datetime
import os
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import re
import math
from tensorflow.keras.layers import (StringLookup,
                                     TextVectorization,
                                     Embedding,
                                     Normalization,
                                     Concatenate,
                                     Flatten,
                                     Dropout,
                                     GlobalAveragePooling1D)

from tensorflow import keras
from tensorflow.keras import layers

# Mix with the embeddings
def get_embeddings_recipe_id(recipe_id: str, embedding_dict: Dict[str, np.array]):
  return embedding_dict[recipe_id]

# generate features dictionaries
def generate_features(df, feature_list, numeric_features):
  dict_features = {}
  for feature in feature_list:
    #print(f"feature: {feature}")
    if feature in numeric_features:
      #print(f"feature numeric: {feature}")
      dict_features[feature] = tf.convert_to_tensor(df[feature].values, tf.float32, name=f"{feature}")
    else:
      dict_features[feature] = tf.convert_to_tensor(df[feature].values, tf.string, name=f"{feature}")
  return dict_features

# cosine similarity layer
class CosineSimilarity(tf.keras.layers.Layer):
  def __init__(self):
    super(CosineSimilarity, self).__init__()

  def build(self, input_shape):
    pass

  def call(self, inputs):
    x, y = inputs
    dot = tf.reduce_sum(x*y, axis=0)
    #print(f"dot: {dot}")
    norm_x = tf.norm(x, axis=1)
    norm_y = tf.norm(y, axis=1)
    #print(f"norm x: {norm_x}")
    #print(f"norm y: {norm_y}")
    sim = tf.math.divide_no_nan(dot, norm_x*norm_y)
    return tf.expand_dims(sim, axis=0)

import string

def replace_bad_characters(text: str):
  new_text = text.lower()
  new_text = new_text.replace('.', '')
  new_text = re.sub(' +', '_', new_text)
  new_text = new_text.replace(';', '_')
  new_text = new_text.replace(',', '_')
  new_text = new_text.replace(' ', '_')
  new_text = new_text.replace('/', '_')
  new_text = new_text.replace('-', '_')
  #new_text = text.translate(str.maketrans('', '', string.punctuation))
  return new_text

# generate datasets from data
def identify_data_types(input_list):
  numeric_features = []
  categorical_features = []
  embedding_list = []
  for feat in input_list:
    size = feat.shape[-1]
    if size > 1:
      embedding_list.append(feat.name)
    else:
      if feat.dtype == tf.string:
        categorical_features.append(feat.name)
      else:
        numeric_features.append(feat.name)
  return numeric_features, categorical_features, embedding_list


def create_model_inputs(feature_names: List[str],
                        numeric_features: List[str],
                        shapes_dict: Dict[str, Tuple] = None):
    inputs = {}
    for feature_name in feature_names:
      shape = ()
      if shapes_dict is not None and feature_name in shapes_dict.keys():
        shape = shapes_dict[feature_name]
      if feature_name in numeric_features:
          inputs[feature_name] = layers.Input(
              name=feature_name, shape=shape, dtype=tf.float32
          )
      else:
          inputs[feature_name] = layers.Input(
              name=feature_name, shape=shape, dtype=tf.string
          )
    return inputs

# fill nan in columns
def check_nans(df_test):
  dict_col_nans = {}
  for col in df_test.columns:
    dict_col_nans[col] = sum(df_test[col].isna())
  return dict_col_nans

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')

def encode_inputs(inputs, categorical_dict, user_features, train: pd.DataFrame, recipes_features=[], context_features=[]):
    encoded_features = []
    user_features_list = []
    recipes_features_list = []
    context_features_list = []
    for feature_name in inputs:
      # check if the input is in the list
      if feature_name in user_features_list or feature_name in recipes_features or context_features in context_features:
        print(f"Processing feature: {feature_name}...")
        if feature_name in categorical_dict:
            embedding_type = categorical_dict.get(feature_name)
            if embedding_type is not None:
                if embedding_type["embedding_type"] == "text":
                    # Vocabulary size and number of words in a sequence.
                    vocab_size = 10000
                    sequence_length = embedding_type.get("max_seq_len", 10)
                    # Use the text vectorization layer to normalize, split, and map strings to
                    # integers. Note that the layer uses the custom standardization defined above.
                    # Set maximum_sequence length as all samples are not of the same length.
                    text_dataset = \
                    tf.data.Dataset.from_tensor_slices(
                        np.expand_dims(train[feature_name].to_numpy().astype(str), -1)
                        )
                    vectorize_layer = layers.TextVectorization(
                        standardize=custom_standardization,
                        max_tokens=vocab_size,
                        output_mode='int',
                        output_sequence_length=10
                    )
                    vectorize_layer.adapt(text_dataset.batch(64))
                    vectorized_text = vectorize_layer(inputs[feature_name])
                    embedding_text = layers.Embedding(input_dim=vocab_size,
                                                output_dim=100,
                                                mask_zero=True)(vectorized_text)
                    flatten_layer = layers.Flatten()
                    encoded_feature = flatten_layer(embedding_text)
                else:
                    vocabulary = categorical_dict[feature_name]["vocabulary"]
                    # Create a lookup to convert string values to an integer indices.
                    # Since we are not using a mask token nor expecting any out of vocabulary
                    # (oov) token, we set mask_token to None and  num_oov_indices to 0.
                    if embedding_type["embedding_type"] == "sparse":
                        use_embedding = False
                    else:
                        use_embedding = True
                    lookup = StringLookup(
                    vocabulary=vocabulary,
                    mask_token=None,
                    num_oov_indices=0,
                    output_mode="int" if use_embedding else "binary",
                    )
                    if use_embedding:
                        # Convert the string input values into integer indices.
                        encoded_feature = lookup(inputs[feature_name])
                        embedding_dims = int(math.sqrt(len(vocabulary)))
                        # Create an embedding layer with the specified dimensions.
                        embedding = layers.Embedding(
                            input_dim=len(vocabulary), output_dim=embedding_dims
                        )
                        # Convert the index values to embedding representations.
                        encoded_feature = embedding(encoded_feature)
                    else:
                        # Convert the string input values into a one hot encoding.
                        encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))
        else:
            # Use the numerical features as-is.
            data = tf.data.Dataset.from_tensor_slices(
                np.expand_dims(train[feature_name].to_numpy().astype(np.float32), -1))
            normalization_layer = Normalization(axis=-1, input_dim=1)
            normalization_layer.adapt(data, steps=40)
            encoded_feature = normalization_layer(tf.expand_dims(inputs[feature_name], -1))
            #encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # create submodels
        if feature_name in user_features:
          user_features_list.append(encoded_feature)
        elif len(recipes_features) > 0 and feature_name in recipes_features:
          recipes_features_list.append(encoded_feature)
        elif len(context_features) > 0:
          context_features_list.append(encoded_feature)
        else:
          pass

    # create submodels
    # user sub model
    user_concat = layers.concatenate(user_features_list)
    user_concat = layers.Dense(units=100, activation="linear",
                                 kernel_regularizer="l2",
                               name="user_embedding")(user_concat)
    # recipes sub model
    if len(recipes_features) > 0:
      recipes_concat = layers.concatenate(recipes_features_list)
      recipes_concat = layers.Dense(units=100, activation="linear",
                                  kernel_regularizer="l2",
                                    name="recipes_embedding")(recipes_concat)
    else:
      recipes_concat = None

    # context sub model
    if len(context_features) > 0:
      context_concat = layers.concatenate(context_features_list)
      context_concat = layers.Dense(units=32, activation="linear",
                                  kernel_regularizer="l2",
                                    name="context_embedding")(context_concat)
    else:
      context_concat = None

    #all_features = layers.concatenate([user_concat, recipes_concat, context_concat])
    #all_features = Concatenate()(encoded_features)
    return user_concat, recipes_concat, context_concat



def encode_features(model_inputs, feature_list,
                    categorical_dict,
                    numerical_features,
                    train: pd.DataFrame,
                    final_layer_size=100,
                    layer_name = 'concat_layer',
                    embedding_list = None,
                    defined_input_shape=False):
  encoded_features_list = []
  for feature_name in feature_list:
    # check if the input is in the list
    print(f"Processing feature: {feature_name}...")
    if feature_name in categorical_dict:
      embedding_type = categorical_dict.get(feature_name)
      if embedding_type is not None:
        if embedding_type["embedding_type"] == "text":
          # Vocabulary size and number of words in a sequence.
          vocab_size = 10000
          sequence_length = embedding_type.get("max_seq_len", 10)
          # Use the text vectorization layer to normalize, split, and map strings to
          # integers. Note that the layer uses the custom standardization defined above.
          # Set maximum_sequence length as all samples are not of the same length.
          text_dataset = \
          tf.data.Dataset.from_tensor_slices(
              np.expand_dims(train[feature_name].to_numpy().astype(str), -1)
              )
          vectorize_layer = layers.TextVectorization(
              standardize=custom_standardization,
              max_tokens=vocab_size,
              output_mode='int',
              output_sequence_length=10
          )
          vectorize_layer.adapt(text_dataset.batch(64))
          vectorized_text = vectorize_layer(model_inputs[feature_name])
          embedding_text = layers.Embedding(input_dim=vocab_size,
                                      output_dim=100,
                                      mask_zero=True)(vectorized_text)
          flatten_layer = layers.Flatten()
          encoded_feature = flatten_layer(embedding_text)
        else:
          vocabulary = categorical_dict[feature_name]["vocabulary"]
          # Create a lookup to convert string values to an integer indices.
          # Since we are not using a mask token nor expecting any out of vocabulary
          # (oov) token, we set mask_token to None and  num_oov_indices to 0.
          if embedding_type["embedding_type"] == "sparse":
              use_embedding = False
          else:
              use_embedding = True
          lookup = StringLookup(
          vocabulary=vocabulary,
          mask_token=None,
          num_oov_indices=0,
          output_mode="int" if use_embedding else "binary",
          )
          if use_embedding:
              # Convert the string input values into integer indices.
              encoded_feature = lookup(model_inputs[feature_name])
              embedding_dims = int(math.sqrt(len(vocabulary)))
              # Create an embedding layer with the specified dimensions.
              embedding = layers.Embedding(
                  input_dim=len(vocabulary), output_dim=embedding_dims
              )
              # Convert the index values to embedding representations.
              encoded_feature = embedding(encoded_feature)
              # Flatten the embedding layer to a 1D tensor.
              if defined_input_shape:
                encoded_feature = layers.Flatten()(encoded_feature)
          else:
              # Convert the string input values into a one hot encoding.
              if defined_input_shape:
                encoded_feature = lookup(model_inputs[feature_name])
              else:
                encoded_feature = lookup(tf.expand_dims(model_inputs[feature_name], -1))
    else:
      # Use the numerical features as-is.
      # data = tf.data.Dataset.from_tensor_slices(
      #     np.expand_dims(train[feature_name].to_numpy().astype(np.float32), -1))
      # normalization_layer = Normalization(input_shape=(1,))
      # normalization_layer.adapt(data, steps=40)
      # encoded_feature = normalization_layer(tf.expand_dims(model_inputs[feature_name], -1))
      if embedding_list is not None:
        if feature_name in embedding_list:
          encoded_feature = model_inputs[feature_name]
        else:
          if defined_input_shape:
            encoded_feature = model_inputs[feature_name]
          else:
            encoded_feature = tf.expand_dims(model_inputs[feature_name], -1)
      else:
        if defined_input_shape:
          encoded_feature = model_inputs[feature_name]
        else:
          encoded_feature = tf.expand_dims(model_inputs[feature_name], -1)
    encoded_features_list.append(encoded_feature)
  all_features = layers.concatenate(encoded_features_list)
  all_features = layers.Dense(units=final_layer_size,
                              activation="linear",
                              kernel_regularizer="l2",
                              name=layer_name)(all_features)
  return all_features




def get_intermediate_model(full_model, layer_idx):
  intermediate_layer_model = tf.keras.Model(inputs=full_model.input,
                                 outputs=full_model.layers[layer_idx].output)
  return intermediate_layer_model

# create model
def create_model_m1(user_layers, food_layers, context_layers, model_inputs, units=[128, 64, 32]):
  # calculate alingning
  cosine_similarity = layers.Dot(axes=1, normalize=True)([user_layers, food_layers])
  fusion_layer = layers.concatenate([user_layers, food_layers, context_layers, cosine_similarity])
  fusion_layer = layers.BatchNormalization()(fusion_layer)
  if units is not None and len(units) > 0:
    # create layers
    for idx, num_units in enumerate(units):
      fusion_layer = layers.Dense(num_units, activation='relu', kernel_regularizer='l2', name=f'fc_layer_{idx}')(fusion_layer)
  outlayer = layers.Dense(1, activation="sigmoid", kernel_regularizer="l2", name="output_0")(fusion_layer)
  model = keras.Model(inputs=model_inputs, outputs=outlayer)
  return model



def plot_model(model: tf.keras.Model, show_shapes=True, rankdir='LR'):
    tf.keras.utils.plot_model(model, show_shapes=show_shapes, rankdir=rankdir)

def generate_dataset_from_pandas(df:pd.DataFrame, numeric_features, feature_columns, embedding_columns, target_col):
  dict_features = generate_features(df, feature_columns, numeric_features)
  dict_embeddings = {}
  for emb_col in embedding_columns:
    dict_embeddings[emb_col] = tf.convert_to_tensor(np.array(df.loc[:, emb_col].tolist()), dtype=tf.float32)
  dict_features.update(dict_embeddings)
  #dict_labels = generate_features(train, TARGET_FEATURE_LABELS, TARGET_FEATURE_LABELS)
  labels = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(df.loc[:, target_col].values, dtype=tf.float32))
  features = tf.data.Dataset.from_tensor_slices(dict_features)
  # embeddings_ds = tf.data.Dataset.from_tensor_slices(dict_features)
  # features_ds = tf.data.Dataset.zip((features, embeddings_ds))
  ds = tf.data.Dataset.zip((features, labels))
  #train_ds = tf.data.Dataset.from_tensor_slices((features_list, labels_list))
  return ds

def prepare_input(features_df, model_inputs):
  dict_features = {}
  for inp in model_inputs:
    if inp.dtype.name == 'string':
      dict_features[inp.name] = tf.convert_to_tensor(features_df[inp.name].values, dtype=tf.string)
    elif inp.dtype.name == 'float32':
      dict_features[inp.name] = tf.convert_to_tensor(features_df[inp.name].values, dtype=tf.float32)
    elif inp.dtype.name == 'float64':
      dict_features[inp.name] = tf.convert_to_tensor(features_df[inp.name].values, dtype=tf.float64)
    elif inp.dtype.name == 'int64':
      dict_features[inp.name] = tf.convert_to_tensor(features_df[inp.name].values, dtype=tf.int64)
    else:
      raise Exception("No recognized data type")
  return dict_features


def train_model(train_ds: tf.data.Dataset, 
                val_ds: tf.data.Dataset, 
                model_context_m1: tf.keras.Model,
                model_name: str = 'training_model_0_cbow'):
    # train the model and save partial results
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)

    checkpoint_path = f'{model_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)
    early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0.03,
                                                    patience=5,
                                                    restore_best_weights=True)
    # This callback will stop the training when there is no improvement in

    history = model_context_m1.fit(train_ds,
                        epochs=50,
                        validation_data=val_ds,
                        validation_steps=500,
                        callbacks=[tensorboard_callback,
                                        cp_callback,
                                        early_callback])
    return history, checkpoint_path, model_context_m1
  
def create_model_inputs(feature_names: List[str],
                        numeric_features: List[str],
                        shapes_dict: Dict[str, Tuple] = None):
    inputs = {}
    for feature_name in feature_names:
      shape = ()
      if shapes_dict is not None and feature_name in shapes_dict.keys():
        shape = shapes_dict[feature_name]
      if feature_name in numeric_features:
          inputs[feature_name] = layers.Input(
              name=feature_name, shape=shape, dtype=tf.float32
          )
      else:
          inputs[feature_name] = layers.Input(
              name=feature_name, shape=shape, dtype=tf.string
          )
    return inputs