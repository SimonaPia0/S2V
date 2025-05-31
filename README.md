# Graph Classification Noisy Labels
Graph classification model developed by using PyTorch Geometric and Pytorch Lightining, focusing on the robusteness over the label noise.

## Project
This project implements a graph neural network model (GNN), based on GINEConv architecture for the graph classification. Pytorch Lightning framework is used for simplify the training and for evaluation.
A key characteristic is the combination of different strategies for label noise handling such as the Generalized Cross Entropy (GCE) loss.

### Characteristics
- GNN (GINEConv) implementation for graph classification
- Usage of structural nodes features (degree, closeness centrality, betweenness centrality, PageRank, clustering coefficient and Laplacian eigenvectors)
- Usage of Generalized Cross Entropy (GCE) loss
- Logger and callback for detailed monitoring and best models saving
- Test set prediction

## Guide

### Project structure
- source/
  - model.py
  - loadData.py
- main.py
- checkpoints/
- submission/
- logs/
- requirements.txt
- zipthefolder.py
- README.md

## Usage

### Training and predictions

Run the main.py script with the following arguments:
'''bash
python main.py --test_path path_to_the_test_folder --train_path path_to_the_train_folder --loss_type gce

Arguments explanation:
- --test_path: path to the test dataset
- --train_path: path to the train dataset
- --loss_type: selection of the loss function type

## Results
The method achieves robust performance across different datasets with varying types of noise. The use of GCE for noise handling, combined with the structure features and the use of GINEConv, ensures that the model generalizes well and produces accurate predictions.