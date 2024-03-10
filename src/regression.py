from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from .parent import Parent

ALGORITHM_NAMES = [
    "Linear Regression",
    "SVR",
    "K-Neighbors Regressor",
    "Random Forest Regressor",
    "Decision Tree Regressor",
]
ALGORITHM_OBJECT = [
    LinearRegression,
    SVR,
    KNeighborsRegressor,
    RandomForestRegressor,
    DecisionTreeRegressor,
]
ALGORITHM_REG = dict(zip(ALGORITHM_NAMES, ALGORITHM_OBJECT))
METRICS_NAMES = [
    "mean_absolute_error",
    "mean_squared_error",
    "mean_absolute_percentage_error",
]
METRICS_OBJECT = [
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
]


class Regression(Parent):
    ALGORITHM = ALGORITHM_REG
    METRICS = dict(zip(METRICS_NAMES, METRICS_OBJECT))
    model_type = "Regression"

    def __str__(self) -> str:
        return "<Regression Object>"

    def find_best_model(self, metric="mean_absolute_percentage_error"):
        """
        Mencari model terbaik berdasarkan nilai metrik evaluasi yang diberikan.

        Parameters:
            metric (str, optional): Nama metrik evaluasi yang akan digunakan untuk menemukan model terbaik.
                                    Jika "all", akan menggunakan total kesalahan untuk semua metrik evaluasi yang tersedia.
                                    Defaultnya adalah "mean_absolute_percentage_error".

        Returns:
            tuple: Tuple yang berisi algoritma terbaik dan nilai metrik terbaik.

        Raises:
            ValueError: Jika nilai metrik evaluasi tidak valid.

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Menemukan model terbaik berdasarkan metrik evaluasi yang diberikan
            best_algorithm, best_metric_value = model.find_best_model(metric="mean_squared_error")
        """

        rest = self.result_compare_models
        for i in rest:
            if metric == "all":
                err = sum(rest[i].values())
            else:
                if metric not in rest[i].keys():
                    raise ValueError(f"{metric} not in metrics")
                err = rest[i][metric]
            rest[i] = err
        best_algo = min(rest, key=rest.get)
        self.best_algorihtm = best_algo
        return best_algo, rest[best_algo]

    def compare_model(self, output="dict"):
        """
        Membandingkan performa model menggunakan berbagai algoritma klasifikasi yang telah ditentukan sebelumnya.

        Parameters:
            output (str, optional): Format output yang diinginkan. Nilai yang diterima adalah "dict" atau nama metrik evaluasi.
                                    Jika "dataframe" atau nama metrik evaluasi tidak diberikan, akan mengembalikan nilai metrik evaluasi yang diminta.
                                    Defaultnya adalah "dict".

        Returns:
            dict or pandas.DataFrame or float: Hasil evaluasi model untuk setiap algoritma klasifikasi.
                                               Jika output adalah "dict", akan mengembalikan kamus yang berisi hasil evaluasi untuk setiap algoritma.
                                               Jika output adalah "dataframe", akan mengembalikan DataFrame Pandas yang berisi hasil evaluasi untuk setiap algoritma.
                                               Jika output adalah nama metrik evaluasi, akan mengembalikan nilai metrik evaluasi tersebut untuk setiap algoritma.

        Raises:
            ValueError: Jika nilai output tidak valid atau jika nama metrik evaluasi tidak ditemukan.

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Membandingkan performa model menggunakan metode compare_model
            result_dict = model.compare_model()
            result_dataframe = model.compare_model(output="dataframe")
            accuracy_dict = model.compare_model(output="accuracy")
        """
        X_train, X_test, y_train, y_test = (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )
        result = {}
        for al in self.ALGORITHM:
            metrics = {}
            alg = self.ALGORITHM[al]
            pred = alg().fit(X_train, y_train).predict(X_test)
            for m, v in self.METRICS.items():
                metrics[m] = round(v(y_true=y_test, y_pred=pred), 2)
            result[al] = metrics

        self.result_compare_models = result
        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "dict":
            return result
        else:
            if output not in self.METRICS.keys():
                raise ValueError("output not in metrics")
            rest = {}
            for i in result:
                rest[i] = result[i][output]
            return rest
