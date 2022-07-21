import logging
from dataclasses import dataclass
from pathlib import Path

import sklearn
from codetiming import Timer
from omegaconf import DictConfig
from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import KFold, GroupKFold, train_test_split, StratifiedKFold
from sklearn.model_selection._validation import _score
from sklearn.neighbors import KNeighborsClassifier


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Integer
from skopt.utils import use_named_args, dump
from torch import nn
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from sklike_torch.torch_wrapper import TorchWrapper, MLP


@dataclass
class Configuration():
    pass

@dataclass
class DataDescription():
    cat_cols: list
    num_cols: list
    id_cols: list
    meta_data: list
    targets: list
    nrow: list
    ncol: list

    pass

class TqdmCallback(tqdm):

    def __call__(self, res):
        super().update()

    def __getstate__(self):
        return []
    def __setstate__(self, state):
        pass

class Experiment():
    """
    Pipeline
        --init
        --add data
        --add model, either add_sk_model or add_torch_model
        --prepare data
        -- define train/test split
        -- define validation method, either define_test_split or define_cv_split
    """

    def __init__(self, log_id="unnamed",seed=42):

        self.seed = seed
        self.log_id = log_id

        self._initialise()


    def _initialise(self):
        np.random.seed(self.seed)

        #setup log directory and logging
        self.log_dir, self.exp_num = self._setup_dir(self.log_id)
        self.logger = self._setup_logger(self.log_dir)

        #data params
        self.data = None #pd.Dataframe containing data
        self.data_desc = None #dataclass with information on our data

        #model params
        self.model = None
        self.model_params = None
        self.model_type = None

        #train/test split
        self.test_splitter = None
        self.stratified = False

        #cross validation split
        self.val_method = None # cv,
        self.cv_splitter = None # class
        self.cv_groups = None
        self.holdout_splitter = None # function

        #pararms from prepare_experiment
        self.task = None
        self.target = None

    def reset(self):
        self._initialise()

    def add_data(self, data, cat_cols=None, num_cols=None, id_cols=None, meta_data=None, targets=None):

        if targets is None:
            targets = [data.columns[-1]]

        if cat_cols is None:
            cat_cols = []

        if id_cols is None:
            data["id"]=data.index.to_numpy()
            id_cols = ["id"]

        if meta_data is None:
            meta_data = []

        if num_cols is None:
            num_cols = [c for c in data.columns if
                        (not c in cat_cols) and (not c in id_cols) and (not c in meta_data) and (not c in targets)]

        nrow, ncol = data.shape

        self.data=data
        self.data_desc = DataDescription(cat_cols=cat_cols, id_cols=id_cols, meta_data=meta_data, targets=targets,
                                         num_cols=num_cols, nrow=nrow, ncol=ncol)

        self.logger.info("Data added")
        return self

    def add_torch_model(self, model,
                        pretrained=False,

                        batch_size=32,
                        epochs=100,
                        gpus=None,
                        log_target="tensorboard",

                        **kwargs #kwargs are parameters specific to the model we are wrapping
                        ):
        """
        Adds a torch model, which is wrapped in our TorchWrapper
        :param model: an nn.Module or pl.BaseModel
        :param model_config:
        :param pretrained: flag for if model is pretrained
        :param batch_size:
        :param epochs:
        :param gpus:
        :param project_name:
        :param run_name:
        :param log_target:
        :param kwargs: parameters specific to model
        :return:
        """

        self.model_params_ind=kwargs
        self.model_type = "torch"
        self.model_params = {'pretrained': pretrained,
                          'batch_size': batch_size,
                          'epochs': epochs,
                          'gpus': gpus,
                          'project_name': self.log_dir,
                          'log_target': log_target,
                          'run_name': self.log_id,
                          'model_params':kwargs}
        class _InnerClass(IdentityLearner):

            def __new__(cls,**kwargs):
                return TorchWrapper(model,**kwargs)

            def fit(self,X,y):
                return self

            def predict(self,X):
                return X
        self.model= _InnerClass
        return self

    def add_sk_model(self, model, **kwargs):
        self.model = model
        self.model_params = kwargs
        self.logger.info("scikit model added")
        self.model_type = "sklearn"
        return self

    def _one_hot_encode(self,data,data_desc):
        cat_names = data_desc.cat_cols

        if len(cat_names)>0:
            cat_data = data[cat_names]

            encoder = OneHotEncoder()
            cat_data =encoder.fit_transform(cat_data)

            data.drop(columns=cat_names)
            cat_names = encoder.get_feature_names_out(cat_names)

            for i,name in enumerate(cat_names):
                data[name] = cat_data[:,i]
            data_desc.cat_cols = cat_names

        return data,data_desc

    def prepare_experiment(self, task = "classification", scoring=None, target=None, one_hot_encode=True, n_bins_target=2):
        """
        Aggregates information from previous steps,transforms data and updates model parameters (if applicable)

        Users may need to overwrite this method to get the functionality they want.
        :param one_hot_encode:
        :return:
        """

        #store params
        self.task = task
        if target == None:
            target = self.data_desc.targets[0]
        self.target = target
        #todo case of unsupervised

        #define our scorers
        if scoring is None:
            if self.task == "classification":
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'

        if callable(scoring):
            self.scorer = scoring
        elif isinstance(scoring, str):
            self.scorer = check_scoring(self.model, scoring)
        else:
            self.scorer = _check_multimetric_scoring(self.model, scoring)

        #check we need to encode target for classification
        if self.task == "classification":
                ty = type(self.data[self.target][0])
                if ty == str:
                    label_enc = LabelEncoder().fit(self.data[self.target])
                    self.data[self.target]=label_enc.transform(self.data[self.target])
                elif ty == float:
                    pass
                    #todo bin into x num bins
                else:
                    pass
        elif type(self.data[self.target][0]) is str:
            self.data[self.target] = LabelEncoder().fit_transform(self.data['Target'])

        if self.model_type == "torch":
            #for pytorch models we need to associate our data info to them
            self.model_params["continuous_columns"] = self.data_desc.num_cols
            self.model_params["categorical_columns"] = self.data_desc.cat_cols
            self.model_params["target"] = [self.target]
            self.model_params["task"] = self.task

        else:
            pass #sklearn models shoudn't need to worry

        #log our experiment setup
        with open(self.log_dir/"experiment_setup.txt","w+") as f:
            f.write("--------------------\n")
            f.write(f"Experiment: {self.log_dir.parent.name}-{self.exp_num}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write("--------------------\n")
            f.write(f"{self.data_desc}\n")
            f.write("--------------------\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Hyperparameters: {self.model_params}\n")
            f.write("--------------------\n")

        if one_hot_encode:
            self.data, self.data_desc = self._one_hot_encode(self.data, self.data_desc)

        self.logger.info("Experiment Setup")
        return self

    def define_test_split(self, test_percentage=0.2, test_indices=None, f=None, custom_splitter=None):
        """
        Reserve a training and test split by the following methods
        Methods should split based on index #todo create id field

        -random percentage (todo stratified by column)
        -fixed indices
        -values
        each method defines a function taking the following parameters
            train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
        and returning a list of length 2,

        :return:
        """
        from functools import partial

        if not custom_splitter is None:
            self.test_splitter = custom_splitter
        elif not f is None:
           # eg  f = lambda x: x["abc"] == "Group 1"
            test = f(self.data).index.to_numpy()
            train = [i for i in self.data.index.to_numpy() if not i in test]

            def _split(array):
                return array[train],array[test]
            self.test_splitter = _split

        elif not test_indices is None:
            def _split(array):
                train = [i for i in array if i not in test_indices]
                test = [i for i in array if i in test_indices]
                return train,test

            self.test_splitter = _split
        else:          #take percentage
            self.test_splitter = partial(train_test_split, test_size=test_percentage)

        self.logger.info("Test split defined")
        return self


    def define_cross_val_split(self, n_folds=5, groups=None, custom_splitter=None,shuffle=True,stratified=None):
        """
        Setup cross validation method
        Adds options for KFold, Grouped KFold or a custom sk-learn splitter class
        :param n_folds:
        :param groups:
        :param custom_splitter:
        :return:
        """
        self.val_method = "cv"
        if not custom_splitter is None:
            self.cv_splitter = custom_splitter
        else:
            if not groups is None:
                self.cv_splitter = GroupKFold(n_folds, groups)
                self.cv_groups = groups
            elif not stratified is None:
                self.cv_splitter = StratifiedKFold(n_folds,shuffle=shuffle)
                self.stratified=True
            else:
                self.cv_splitter = KFold(n_folds,shuffle=shuffle)
        self.logger.info("Cross Validation Setup")
        return self

    def validate(self,log):
        if self.val_method == "cv":
            return self.cross_validate(log)
        else:
            return self.holdout_loop(log)

    def _log_preds(self, preds, trues, score):
        trues = trues.flatten()
        preds = preds.flatten()

        m, b = np.polyfit(trues, preds, 1)
        fig, ax = plt.subplots()

        ls = np.linspace(min(trues), max(trues))
        ax.plot(ls, ls * m + b, color="black", label=r"$\hat{y}$ = " + f"{m:.4f}y + {b:.4f}")
        ax.scatter(x=trues, y=preds, label=r"$R^2$" + f"={score:.4f}")

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend(bbox_to_anchor=(0.5, 1))

        # todo save predictions by id
        fig.savefig(self.log_dir / f"cross_val.png")

    def cross_validate(self,log=True, return_model_states=False):
        """
        We only allow a single scorer so that this can be used as a function for search
        Scoring copied as much as possible from sklearn cross_validate
        :param scorer: an sklearn scoring metric. default is accuracy_score for classification and -mse for regression
        :param log:
        :return:
        """

        if self.cv_splitter is None:
            raise Exception("Cross Validation splitter not defined")

        if return_model_states:
            states=[]

        #take training data with test_splitter
        train_ind,_ = self.test_splitter(self.data.index.to_numpy())
        x_ind = [i for i,c in enumerate(self.data.columns) if c in self.data_desc.num_cols]
        y_ind = [i for i, c in enumerate(self.data.columns) if c == self.target][0]

        #handle different cv cases
        if self.stratified:
            cv_gen = self.cv_splitter.split(self.data.index[train_ind],data[y_ind][train_ind])
        elif  self.cv_groups is not None:
            group_ind = [i for i, c in enumerate(self.data.columns) if c == self.cv_groups]
            cv_gen =self.cv_splitter.split(self.data.index.iloc[train_ind,:], groups=self.data.index.iloc[train_ind,group_ind])
        else:
            cv_gen = self.cv_splitter.split(self.data.index[train_ind])

        preds = None
        ys = None
        for fold, (inds1, inds2) in enumerate(cv_gen):
            X_train = self.data.iloc[inds1, x_ind] #keep X as pandas
            y_train = self.data.iloc[inds1, y_ind].to_numpy()

            model_ = self.model(**self.model_params)
            model_.fit(X_train, y_train)

            X_test = self.data.iloc[inds2, x_ind]
            y_test = self.data.iloc[inds2, y_ind]

            if self.model_type == "torch":
                pred = model_.predict(X_test, y_test) #why does PYTORCHTABULAR MAKE ME HAVE TO DO THIS ARGHHHH
            else:
                pred = model_.predict(X_test)

            if preds is None:
                preds = pred
                ys = y_test.to_numpy().flatten()
            else:
                preds = np.concatenate((preds, pred), axis=0)
                ys = np.concatenate((ys, y_test), axis=0)

        #calculate scores
        scores = _score(IdentityLearner(), preds, ys, self.scorer, error_score="raise")

        if log:
            self._log_preds(preds,ys,scores)
            self.logger.info("Cross Validation Completed")

        if return_model_states:
            return scores, states
        else:
            return scores

    def define_holdout_split(self, percentage=0.2, test_indices=None, custom_splitter=None):
        self.val_method = "holdout"
        #todo
        return self

    def _setup_logger(self,log_dir,logging_level= logging.INFO):
        #kindly taken from https://stackoverflow.com/a/13733863
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler("{0}/{1}.log".format(log_dir, "log"))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        rootLogger.setLevel(logging_level)
        return rootLogger

    def _setup_dir(self,log_id):
        log_dir = Path("Experiments")/log_id
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        children = sorted([c for c in log_dir.iterdir() if c.is_dir()], key= lambda x: int(x.name))
        if len(children) == 0:
            num = 0
        else:
            num = str(int(children[-1].name)+1)

        log_dir = log_dir/str(num)
        log_dir.mkdir()
        return log_dir,num

    def cv_loop(self):
        pass

    def holdout_loop(self):
        pass

    def run(self):
        if self.val_method == "cv":
            return self.cv_loop()
        else:
            return self.holdout_loop()

    def update_params(self,**params):
        for k,v in params.items():
            self.model_params[k]=v

    def objective(self,**params):
        self.update_params(**params)
        #todo make this not have side effects eg by calling original model
        return -self.cross_validate(log=False)

    def bayesian_optimise(self,space,n_calls=50,random_state=0):
        obj = use_named_args(space)(self.objective)
        return gp_minimize(obj,space,n_calls=n_calls,random_state=random_state,callback=TqdmCallback(total=n_calls))

    def random_optimise(self,space,n_calls=50, random_state=0):
        obj = use_named_args(space)(self.objective)
        return gp_minimize(obj,space,n_calls=n_calls,n_initial_points=n_calls,initial_point_generator='random',
                             random_state=random_state,callback=TqdmCallback(total=n_calls))

    def grid_optimise(self,space,n_calls=50, random_state=0):
        obj = use_named_args(space)(self.objective)
        return gp_minimize(obj,space,n_calls=n_calls,n_initial_points=n_calls,initial_point_generator='grid',
                             random_state=random_state,callback=TqdmCallback(total=n_calls))

    @Timer()
    def search(self, space, strategy="bayesian", n_calls=50, random_state=5, search_name=None):

        #setup directory for search
        if search_name is None:
            self.log_dir = self.log_dir/"search"
        else:
            self.log_dir = self.log_dir/search_name
        if not self.log_dir.exists():
            self.log_dir.mkdir()


        if random_state is None:
            #todo use existing random state, while providing option for controlled
            pass

        if strategy == "bayesian":
            result = self.bayesian_optimise(space, n_calls=n_calls, random_state=random_state)
        elif strategy == "grid":
            result = self.grid_optimise(space, n_calls=n_calls, random_state=random_state)
        elif strategy == "random":
            result = self.random_optimise(space, n_calls=n_calls, random_state=random_state)

        #set parameters of model
        params = {dim.name: result['x'][i] for i, dim in enumerate(space)}
        self.update_params(**params)

        #save model and search
        del result.specs['args']['func'] #spaghetti code to not throw an error as the objective function is unserialisable
        dump(result, self.log_dir/'search.pkl')
        dump(self.model, self.log_dir/'model.joblib')

        #plot search results
        fig, ax = plt.subplots()
        plot_convergence(result, ax=ax)
        fig.savefig(self.log_dir/"convergence_plot.png")

        #log results
        self.logger.info("Search Complete")
        self.logger.info(f'Best model had a Score of {result.fun:.4f}')
        self.logger.info(f'Setting parameters as: {params}')

        #revert log dir - note this is messy, in future we will pass paths down (optionally) or use a try-catch-finally
        self.log_dir = self.log_dir.parent

        return self.model(params)

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()


class IdentityLearner(BaseEstimator):
    """
    A class that makes the prediction X -> X
    """
    def fit(self,X,y):
        return self

    def predict(self,X):
        return X


if __name__ == "__main__":
    import pandas as pd
    e1 = Experiment()
    model = MLP
    task = "classification"
    params = {'n_inputs': 4, 'n_classes': 3, 'widths': [4, 4, 4, 4], 'n_layer': 4, 'type': task}
    data = pd.read_csv(Path.cwd().parents[0]/"data"/"iris.csv")

    e1 = e1.add_torch_model(model, **params).add_data(
        data).prepare_experiment(task=task).define_test_split().define_cross_val_split()
    mse = e1.cross_validate(log=True)
    print(mse)
