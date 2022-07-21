# sklike-torch - An experiment framework for Tabular Data

As I was researching my Masters I kept running into three key problems:
* How do I keep track of experiments? 
* How do I split my data validate my experiments? 
* How do I train and make predictions with neural-networks.  

Great solutions to these problems exist but they tend to differ across the scikit-learn and PyTorch platforms. The 
core philisophy of this library is to build on these existing solutions while providing a consistent scikit-learn type
interface. This is currently only for tabular data and local execution. 

Core Features are:

* A scikit-learn compatability layer for PyTorch models. Call fit(X,y,**params), transform(X) and predict(X). The 
PyTorch-Tabular/PyTorch Lightning 
* A modular experiment framework. Easy to extend by overwriting components or adding your own functions
* Experiment tracking with options for Tensorboard and Weights & Biases. 
* Hyperparameter search with scikit-optimise. Support for  search libraries are todo. 

Examples of things you can do with this library include:
* Train a PyTorch model then add it as a feature extractor to a  scikit-learn pipeline.
* Perform cross-validation then return an ensemble of each fold. 

## Future Developments

This library is currently early in development. Once it reaches a mature stage I plan to break off the torch/scikit 
compatability layer into its own library.

Development is currently limited by the dependency on PyTorch-Tabular. PyTorch-Tabular is 80% excellent but the
remaining 20% has caused a few headaches. Plans may change depending on this library develops.

The current to do list is:
* Exhaustive testing
* Documentation 
* Fixing several logging incompatabilities between library. Add an option to disable logging for PyTorch models.
how this library develops



