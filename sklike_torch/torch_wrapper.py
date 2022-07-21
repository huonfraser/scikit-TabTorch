from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.config import ModelConfig, DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from pytorch_tabular.models import BaseModel

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import ModuleList

from sklike_torch.utils import combineXy



class PyTabWrapper(BaseModel):
    def __init__(
            self,
            config: DictConfig,
            model: nn.Module,
            model_params= {},
            **kwargs
    ):
        """
        Wrap a Pytorch Model into a PyTorchTabular BaseModel

        All the messiness is handled by careful intilisasion. In the future things may be changed to incorporate new features
        :param model:
        :param config:
        :param kwargs:
        """
        self.model_params = model_params
        self.model_callable = None
        self.model_instance = None

        if callable(model): #model is uninitialised
            self.model_callable = model
        else:
            self.model_instance = model
        super().__init__(config,**kwargs)

    def _build_network(self):
        if not self.model_callable is None: # model is unitiliased
            self.model_instance = self.model_callable(**self.model_params)
        else: #model has been predefined, leave as is
            pass
            #todo we need a method to update params and reset model that way
            #self.model.__class__(**self.hparams.model_args)
    def forward(self, x):
        y_hat = self.model_instance.forward(x['continuous'])
        return {'logits': y_hat}


class TorchWrapper(BaseEstimator):
    """
    A compatability layer providing PyTorch nn.Module classes with the sklearn interface.
    Pytorch-Tabular is used as an intermediate layer to abstract away boilerplate.

    PyTorch-Tabular requires passing in an uninitialised/callable class

    I'm working on supporting for wrapping pretrained models or loading model states from file
    Also working on supporting PyTorchTabular Models

    Pass in your model and accompanying hyperparameters to let Pytorch-Tabular do the work, or wrap a pretrained model
    Functions defined currently are:
    -transform: extract features
    -fit:
    -predict
    """

    def __init__(self,
                 model: nn.Module,
                 model_params={},
                 model_config=None,
                 pretrained=False,
                 task="classification",

                 target=['Target'],
                 continuous_columns=["feature_1","feature_2"],
                 categorical_columns=["cat_feature_1","cat_feature_2"],

                 batch_size=32,
                 epochs=100,
                 gpus=None,

                 project_name="TestProject",
                 run_name="test_run",
                 log_target="tensorboard", #["tensorboard","wandb"
                 **kwargs
                 ):
        """
        Optional parameters are hyperparameters for training and defining a model
        :param model: Three options. Torch nn.Module inisialised, Torch nn.Module unisialised, pytorch-tabular BaseModel uninitialised
        """
        self.tabular_model = None #Our PyTorch Tabular Model
        self.model_params=model_params
        self.pretrained = pretrained #flag for if our inputted model is pretrained

        self.task = task
        #our passed in data parameters
        #todo update these optionally in fit
        self.data_config = DataConfig(target=target,
                                      continuous_cols=continuous_columns,
                                      categorical_cols=categorical_columns)

        #if we didn't design a custom model config use the default one
        if model_config is None:
            model_config = ModelConfig(task=task)

        self.model_config = model_config

        self.trainer_config = TrainerConfig(batch_size=batch_size, max_epochs=epochs, gpus=gpus)
        self.optimizer_config = OptimizerConfig()
        self.experiment_config = ExperimentConfig(project_name=project_name, run_name=run_name,
                                             log_target=log_target)
        self.data_config = DataConfig(target=target,
                                 continuous_cols = continuous_columns,
                                 categorical_cols = categorical_columns)


        if self.pretrained: #if not pretrained initialise during fit
            self.tabular_model = TabularModel(
                data_config=self.data_config,
                model_config=self.model_config,
                optimizer_config=self.optimizer_config,
                trainer_config=self.trainer_config,
                experiment_config=self.experiment_config,
                model_callable=self.tabtorch_model
            )

        if type(model) is BaseModel: #PyTorch tabular model
            self.tabtorch_model = BaseModel
        else: #torch.nn.module
            self.tabtorch_model = lambda config, **kwargs, : PyTabWrapper(config, model, model_params=self.model_params, **kwargs)

            # PartialClass

    def update_params(self, **params):
            for k, v in params.items():
                self.model_params[k] = v

    def fit(self, X, y,**kwargs):
        """
        Fit as per sklearn. May be some headaches here with models of different types
        :param X:
        :param y:
        :param kwargs:
        :return:
        """
        #todo how to handle metadata
        #todo infer data from X
        data = combineXy(X,y)


        self.tabular_model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
            experiment_config=self.experiment_config,
            model_callable=self.tabtorch_model
        )

        self.tabular_model.fit(data)
        return self

    def transform(self, X, y=None):
        """
        Use the PyTorch Tabular DeepFeatureExtractor to extract features
        :param X:
        :param y:
        :return:
        """
        data = combineXy(X, y)
        dt = DeepFeatureExtractor(self.tabtorch_model)
        return dt.fit_transform(data)

    def predict(self,X,y=None):
        """
        :param X:
        :return:
        """
        data = combineXy(X, y=y)
        pred = self.tabular_model.predict(data)
        if self.task == "classification":
            return pred["prediction"].to_numpy().flatten()
        else: #regression
            return pred["Target_prediction"].to_numpy().flatten()

    def predict_proba(self,X):
        pass
        #todo

class TorchRegressor(TorchWrapper, RegressorMixin):
    pass

class TorchClassifier(TorchWrapper, ClassifierMixin):
    pass

class MLP(nn.Module):
    """
    A simple implementation of a Multilayer Perceptron model. Used for testing and debugging purposes
    """

    def __init__(self, n_inputs=2, n_layer=2, n_classes=2, widths=[2, 2], act=nn.ReLU,type="classification"):
        super(MLP, self).__init__()
        layers = OrderedDict()
        layers["input"] = nn.Linear(n_inputs,widths[0])
        layers["input_act"] = act()

        for i in range(1,n_layer):
            layers[f"input_{i}"] = nn.Linear(widths[i-1], widths[i])
            layers[f"act_{i}"] = act()

        if type == "classification":
            layers["head1"] = nn.Linear(widths[-1],n_classes)
            layers["head2"] = nn.Softmax(0)
        else:
            layers["head"] = nn.Linear(widths[-1],1)

        self.network = nn.Sequential(layers)

    def forward(self, X):
        val = self.network.forward(X)
        return val

if __name__=="__main__":
    model = MLP
    task = "regression"
    wrapper = TorchWrapper(model, model_params={'n_inputs':4,'n_classes':3,'widths':[4,4,4,4],'n_layer':4,'type':task}
                           ,categorical_columns=[],
                           continuous_columns=['0','1','2','3'],
                           task=task,
                           epochs=10,
                           project_name="Experiments/Test",
                           run_name="test_run/0/",
                           log_target="tensorboard",
                           )
    data = pd.read_csv(Path.cwd().parents[0]/"data"/"iris.csv")
    y = LabelEncoder().fit_transform(data['Target'])
    X = data.drop(columns=['Target'])
    wrapper.fit(X,y)
    wrapper.predict(X)

