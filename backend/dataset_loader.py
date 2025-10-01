import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer
from sklearn.datasets import fetch_california_housing, make_blobs, make_classification, make_regression
import logging

logger = logging.getLogger(__name__)


def load_dataset(dataset_name):
    """
    加载数据集，返回 (X, y) 的numpy数组格式
    """
    try:
        if dataset_name == "iris":
            data = load_iris()
        elif dataset_name == "diabetes":
            data = load_diabetes()
        elif dataset_name == "california_housing":
            data = fetch_california_housing()
        elif dataset_name == "digits":
            data = load_digits()
        elif dataset_name == "wine":
            data = load_wine()
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
        elif dataset_name == "blobs":
            X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
            return X, y
        elif dataset_name == "classification":
            X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
            return X, y
        elif dataset_name == "regression":
            X, y = make_regression(n_samples=200, n_features=2, noise=0.1, random_state=42)
            return X, y
        else:
            raise ValueError(f"未知数据集: {dataset_name}")

        X = data.data
        y = data.target

        logger.info(f"加载数据集: {dataset_name}, 形状: X={X.shape}, y={y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"数据集加载失败 {dataset_name}: {str(e)}")
        raise