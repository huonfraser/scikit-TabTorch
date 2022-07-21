import shutil
from pathlib import Path
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from skopt.space import Integer

from sklike_torch import experiments, torch_wrapper
from sklike_torch.experiments import Experiment

def load_iris():
    return pd.read_csv(Path("data") / "iris.csv")
def clear_tests():
    #todo make this a decorator
    path = Path("Experiments")/"Test"
    if path.exists():
        shutil.rmtree(path)

def test_log_dir():
    """Test that creating correct directories"""
    clear_tests()

    # delete existing directories
    if (Path("Experiments")/"Test").exists():
        shutil.rmtree(Path("Experiments") / "Test")


    e1 = experiments.Experiment(log_id="test")
    e1.close()
    assert e1.log_dir == Path("Experiments")/"Test"/"0"
    e2 = experiments.Experiment(log_id="test")
    e2.close()
    assert e2.log_dir == Path("Experiments")/"Test"/"1"

    clear_tests()


def test_add_data():
    clear_tests()

    e1 = experiments.Experiment()
    iris = load_iris()
    e1 = e1.add_data(iris)
    dd = e1. data_desc

    assert e1.data is not None
    assert dd.nrow == 150
    assert dd.ncol == 6
    assert dd.cat_cols == []
    assert dd.num_cols == ["0","1","2","3"]
    assert dd.id_cols == ["id"]
    assert dd.meta_data == []
    assert dd.targets == ["Target"]
    assert len(dd.cat_cols)+len(dd.num_cols)+len(dd.id_cols)+len(dd.meta_data)+len(dd.targets) == dd.ncol

    e1.close()
    clear_tests()


def test_add_model():
    clear_tests()

    e1 = experiments.Experiment()
    model = KNeighborsClassifier
    params = {'n_neighbors':5, 'weights':'uniform'}

    e1 = e1.add_sk_model(model,**params)
    e1.close()

    clear_tests()

def test_sk_class():
    clear_tests()
    e1 = Experiment()
    model = KNeighborsClassifier
    params = {'n_neighbors': 5, 'weights': 'uniform'}
    data = load_iris()

    e1 = e1.add_sk_model(model, **params).add_data(data).prepare_experiment().define_test_split().define_cross_val_split()
    mse = e1.cross_validate(log=True)
    clear_tests()

def test_sk_regr():
    clear_tests()
    e1 = Experiment()
    model = KNeighborsRegressor
    params = {'n_neighbors': 5, 'weights': 'uniform'}
    data = load_iris()

    e1 = e1.add_sk_model(model, **params).add_data(data).prepare_experiment(task="regression").define_test_split().define_cross_val_split()
    mse = e1.cross_validate(log=True)
    clear_tests()

def test_torch_class():
    clear_tests()
    e1 = Experiment()
    model = torch_wrapper.MLP
    task = "classification"
    params ={'n_inputs':4,'n_classes':3,'widths':[4,4,4,4],'n_layer':4,'type':task}
    data = load_iris()

    e1 = e1.add_torch_model(model,epochs=2, **params).add_data(data).prepare_experiment().define_test_split().define_cross_val_split()
    mse = e1.cross_validate(log=True)
    clear_tests()

def test_torch_regr():
    clear_tests()
    e1 = Experiment()
    model = torch_wrapper.MLP
    task = "regression"
    params = {'n_inputs': 4, 'n_classes': 3, 'widths': [4, 4, 4, 4], 'n_layer': 4, 'type': task}
    data = load_iris()

    e1 = e1.add_torch_model(model, epochs=2, **params).add_data(data).prepare_experiment(task=task).define_test_split().define_cross_val_split()
    mse = e1.cross_validate(log=True)
    clear_tests()
