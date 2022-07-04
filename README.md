# sklike-torch

sklike-torch provides a high-level api to use pytorch neural-networks alongside sklearn-models.

Core features are:
* An sklearn-compatability layer for pytorch models. Call fit(x,y),transform(x) and predict(x)
* An improved cross-validation loop
* Experiment tracking


## Use pytorch modules like they're sklearn models

Sklearn only provides a simple neural-network models while pytorch is famous for providing a fairly low-level api. Combining the two requires juggling between tensors and ndarrays, torch Ddatasets and pandas Dataframes. Pytorch-tabular comes close to briding this gap and we build upon this package to give an api that combines pytorch and sklearn models seamlessly. 

## Improved cross-validation loop 

Traditional cross-validation metrics average results from each fold. Our implementation concatenates test set predictions from each fold and calcualtes metrics based on the combined set of predictions. This gives a better idea of performance for cases with uneven sized folds or for non-linear metrics like R^2 and is particurly effective for cross-validation methods where each instance is tested on once. Models trained independantly on each fold can be ensembled.


## Weights and Bias Integration

