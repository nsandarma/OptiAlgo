from src.parent import Parent
import pandas as pd

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
    MeanShift,
    DBSCAN,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from collections import defaultdict

ALGORITHM_NAMES = [
    "KMeans",
    "Mini Batch KMeans",
    "Spectral Clustering",
    "Gaussian Mixture",
    "KMedoids",
]

ALGORITHM_OBJECT = [
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
    GaussianMixture,
    KMedoids,
]

METRIC_NAMES = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
METRIC_OBJECTS = [silhouette_score, davies_bouldin_score, calinski_harabasz_score]


class Clustering(Parent):
    ALGORITHM = dict(zip(ALGORITHM_NAMES, ALGORITHM_OBJECT))
    METRICS = dict(zip(METRIC_NAMES, METRIC_OBJECTS))
    model_type = "Clustering"

    def __str__(self):
        return "<Clustering Object>"

    def fit(self, data: pd.DataFrame, features, PCA=False):
        X = pd.get_dummies(data[features]).values
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_transform = scaler.transform(X)
        self.X = X
        if PCA == True:
            pass

        self.X_transform = X_transform
        self.X_test = None
        self.y_test = None
        self.X_train = X_transform
        self.y_train = None

        return self

    def find_best_model(self):
        if not hasattr(self, "result_compare_models"):
            return "None"
        result = self.result_compare_models
        silhouette_values = {
            model: metrics["Silhouette"] for model, metrics in result.items()
        }
        best_algorithm = max(silhouette_values, key=silhouette_values.get)
        self.best_algorithm = best_algorithm
        return best_algorithm, silhouette_values[best_algorithm]

    def compare_model(self, n_clusters, output="dict"):
        result = {}
        for i in self.ALGORITHM:
            al = self.ALGORITHM[i]
            metrics = {}
            for j in self.METRICS:
                label = al(n_clusters).fit_predict(self.X_transform)
                score = self.METRICS[j](self.X_transform, label)
                metrics[j] = score
            result[i] = metrics
        self.result_compare_models = result
        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "only_silhouette":
            rest = {}
            for i in result:
                rest[i] = result[i]["Silhouette"]
            return rest
        else:
            return result
