import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklike_torch import torch_wrapper


def clear_tests():
    #todo make this a decorator
    path = Path("Experiments")/"Test"
    if path.exists():
        shutil.rmtree(path)

    path = Path("saved_models/")
    if path.exists():
        shutil.rmtree(path)

def test_0():
    clear_tests()
    model = torch_wrapper.MLP
    task = "classification"
    wrapper = torch_wrapper.TorchWrapper(model, model_params={'n_inputs': 4, 'n_classes': 3, 'widths': [4,4,4,4],'n_layer': 4, 'type': task},
                           categorical_columns=[],
                           continuous_columns=['0','1','2','3'],
                           task=task,
                           epochs=10,
                           project_name="Experiments/Test",
                           run_name="test_run/0/",
                           log_target="tensorboard",
                           )
    data = pd.read_csv(Path("data")/"iris.csv")
    y = LabelEncoder().fit_transform(data['Target'])
    X = data.drop(columns=['Target'])
    wrapper.fit(X,y)
    clear_tests()
