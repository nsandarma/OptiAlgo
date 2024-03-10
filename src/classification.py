import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
)
from sklearn.preprocessing import MinMaxScaler

# Import Algorithm Module from sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from collections import defaultdict


from .parent import Parent


ALGORITHM_NAMES = [
    "Naive Bayes",
    "K-Nearest Neighbor",
    "SVM",
    "Logistic Regression",
    "Random Forest",
    "Decision Tree Classifier",
]
ALGORITHM_OBJECT = [
    MultinomialNB,
    KNeighborsClassifier,
    SVC,
    LogisticRegression,
    RandomForestClassifier,
    DecisionTreeClassifier,
]
ALGORITHM_CLF = dict(zip(ALGORITHM_NAMES, ALGORITHM_OBJECT))


class Classification(Parent):
    ALGORITHM = ALGORITHM_CLF
    METRICS = ["Accuracy", "Precision", "Recall", "F1 Score"]
    model_type = "Classification"

    def __str__(self) -> str:
        return "<Classificaton Object>"

    def find_best_model(self):
        """
        Mencari model terbaik berdasarkan hasil perbandingan model yang telah disimpan sebelumnya.

        Returns:
            tuple: Tuple yang berisi algoritma terbaik dan akurasi terbaik.
                   Jika tidak ada hasil perbandingan model yang tersedia, akan mengembalikan "None".

        Example:
            # Inisialisasi objek model
            model = Classicication()

            # Menemukan model terbaik
            best_algorithm, best_accuracy = model.find_best_model()
        """
        if not hasattr(self, "result_compare_models"):
            return "None"
        result = self.result_compare_models
        accuracys = defaultdict(float)
        for algo, report in result.items():
            accuracy = report["weighted avg"]["f1-score"]
            accuracys[algo] = accuracy
        best_algorithm = max(accuracys, key=accuracys.get)
        best_accuracy = accuracys[best_algorithm]
        self.best_algorihtm = best_algorithm
        return best_algorithm, best_accuracy

    def compare_model(self, test_size=None, output="dict"):
        """
            Membandingkan performa model menggunakan berbagai algoritma klasifikasi yang telah ditentukan sebelumnya.

        Parameters:
            test_size (float, optional): Persentase ukuran data uji. Defaultnya adalah None, yang akan menggunakan nilai yang telah diatur sebelumnya dalam `split_data`.
            output (str, optional): Format output yang diinginkan. Nilai yang diterima adalah "dict", "dataframe", atau "only_accuracy". Defaultnya adalah "dict".

        Returns:
            dict, pandas.DataFrame, or dict: Hasil evaluasi model untuk setiap algoritma klasifikasi, sesuai dengan format output yang diminta.

            Example:
            # Inisialisasi objek model
            model = Classicication()

            # Membandingkan performa model menggunakan metode compare_model
            result_dict = model.compare_model()
            result_dataframe = model.compare_model(output="dataframe")
            accuracy_dict = model.compare_model(output="only_accuracy")
        """
        if test_size is None:
            test_size = self.split_data

        X_train, X_test, y_train, y_test = (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )
        result = {}
        for al in self.ALGORITHM:
            alg = self.ALGORITHM[al]
            pred = alg().fit(X_train, y_train).predict(X_test)
            result[al] = classification_report(
                y_pred=pred, y_true=y_test, output_dict=True, zero_division=0.0
            )
        self.result_compare_models = result
        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "only_accuracy":
            rest = {}
            for i in result:
                rest[i] = result[i]["accuracy"]
            return rest
        else:
            return result
