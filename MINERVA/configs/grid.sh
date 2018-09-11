#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/grid"
vocab_dir="datasets/data_preprocessed/grid/vocab"
total_iterations=50
path_length=3
hidden_size=25
embedding_size=25
batch_size=64
beta=0.07
Lambda=0.0
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/grid"
load_model=0
model_load_dir="null"
nell_evaluation=0
