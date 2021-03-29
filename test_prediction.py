import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from machine_learning import train_model, export_model


def test_inference_sample():
    model = train_model()
    X, _ = datasets.load_digits(return_X_y=True)
    prediction = model.predict(X[0:1])[0]
    assert prediction == 0

def test_inference_batch():
    model = train_model()
    X, _ = datasets.load_digits(return_X_y=True)
    predictions = model.predict(X[0:100])
    assert np.all(predictions < 10)
