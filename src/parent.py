from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


class Parent(object):
    def __str__(self) -> str:
        return "<OptiAlgo>"

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        features: list,
        norm=True,
        split_data=0.2,
        all_data_train=False,
    ):
        """
        Melatih model dengan data yang diberikan.

        Parameters:
            data (pd.DataFrame): Data yang akan digunakan untuk melatih model.
            target (str): Nama kolom yang merupakan target prediksi.
            features (list): Daftar nama kolom yang akan digunakan sebagai fitur.
            norm (bool, optional): Jika True, fitur-fitur akan dinormalisasi menggunakan MinMaxScaler. Defaultnya adalah True.
            split_data (float, optional): Persentase data yang akan digunakan sebagai data uji jika `all_data_train` adalah False. Defaultnya adalah 0.2.
            all_data_train (bool, optional): Jika True, semua data akan digunakan sebagai data latih tanpa pemisahan. Defaultnya adalah False.

        Returns:
            self: Instance dari objek model yang telah dilatih.

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Melatih model dengan data yang diberikan
            model.fit(data=my_data, target="label", features=["feature1", "feature2"], norm=True, split_data=0.2, all_data_train=False)
        """

        # Memasukkan data, fitur, dan target ke dalam objek model
        self.data = data
        self.X = pd.get_dummies(data[features]).values
        self.y = data[target].values

        # Normalisasi fitur jika diminta
        if norm:
            scaler = MinMaxScaler()
            scaler.fit(self.X)
            self.X_transform = scaler.transform(self.X)
            self.scaler = scaler
            X = self.X_transform
        else:
            self.X_transform = None
            X = self.X

        # Memisahkan data menjadi data latih dan data uji
        if not all_data_train:
            X_train, X_test, y_train, y_test = train_test_split(
                X, self.y, test_size=split_data, random_state=42
            )
        else:
            X_train, y_train = self.X, self.y
            X_test, y_test = None, None

        # Menyimpan data latih dan data uji
        self.X_train, self.y_train, self.X_test, self.y_test = (
            X_train,
            y_train,
            X_test,
            y_test,
        )

        # Menyimpan parameter dan persentase pemisahan data
        self.params = None
        self.split_data = split_data

        return self

    def score(self, y_test, y_pred):
        """
        Menghitung skor evaluasi model berdasarkan prediksi dan nilai sebenarnya.

        Parameters:
            y_test (array-like): Nilai target sebenarnya.
            y_pred (array-like): Nilai yang diprediksi oleh model.

        Returns:
            dict: Skor evaluasi model. Untuk model klasifikasi, akan mengembalikan laporan klasifikasi dalam bentuk kamus.
                  Untuk model regresi, akan mengembalikan metrik evaluasi yang terdiri dari MAE (Mean Absolute Error),
                  MAPE (Mean Absolute Percentage Error), dan MSE (Mean Squared Error).

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Membuat prediksi dengan model
            y_pred = model.predict(X_test)

            # Menghitung skor evaluasi model
            evaluation_scores = model.score(y_test, y_pred)
        """
        if self.model_type == "Classification":
            return classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
        elif self.model_type == "Regression":
            return {
                "MAE": mean_absolute_error(y_pred=y_pred, y_true=y_test),
                "MAPE": mean_absolute_percentage_error(y_pred=y_pred, y_true=y_test),
                "MSE": mean_squared_error(y_pred=y_pred, y_true=y_test),
            }

    def set_model(self, algo_name):
        """
        Mengatur model dengan algoritma yang telah ditentukan.

        Parameters:
            algo_name (str): Nama algoritma yang akan digunakan untuk model.

        Returns:
            self: Instance dari objek model yang telah diatur dengan model baru.

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Mengatur model dengan algoritma yang ditentukan
            model.set_model("RandomForestClassifier")
        """
        try:
            # Memeriksa apakah algoritma tersedia dalam daftar yang telah ditentukan
            if algo_name in self.ALGORITHM:
                # Menginisialisasi model dengan algoritma yang dipilih
                model_instance = self.ALGORITHM[algo_name]
                model = model_instance().fit(self.X_train, self.y_train)
                self.model = (algo_name, model)
                return self
            else:
                raise ValueError("Algorithm not found in the list.")
        except Exception as e:
            print("Error:", e)
            return self

    def hyperparameters(self, params):
        """
        Menyetel hyperparameter model dan mengembalikan laporan klasifikasi.

        Parameters:
            params (dict): Hyperparameter yang akan disetel untuk model.

        Returns:
            dict: Laporan klasifikasi berdasarkan prediksi menggunakan model dengan hyperparameter yang telah disetel.

        Raises:
            ValueError: Jika model tidak ditemukan atau jika ada parameter yang dibutuhkan yang tidak disediakan.

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Menyetel hyperparameter model dan mengembalikan laporan klasifikasi
            hyperparameter_report = model.hyperparameters(params={"n_estimators": 100, "max_depth": 5})
        """
        # Memeriksa apakah model telah diatur sebelumnya
        if not hasattr(self, "model"):
            raise ValueError("Model algorithm not found")

        # Mendapatkan nama algoritma dari model yang telah diatur
        algo_name = self.model[0]

        # Memeriksa apakah algoritma tersedia dalam daftar yang telah ditentukan
        if algo_name not in self.ALGORITHM:
            raise ValueError("Algorithm '{}' not found in the list.".format(algo_name))

        # Mendapatkan model dari daftar algoritma
        model_instance = self.model[1]

        # Memeriksa parameter yang dibutuhkan oleh model
        params_required = list(model_instance.get_params().keys())
        for key in params:
            if key not in params_required:
                raise ValueError(
                    "Parameter '{}' is not required for this model.".format(key)
                )

        # Menyetel hyperparameter model
        model = self.model[1]
        model = model.set_params(**params)

        # Menyimpan model yang telah disetel
        self.model = algo_name, model

        # Melakukan prediksi menggunakan model yang telah disetel
        pred = model.predict(self.X_test)

        # Mengembalikan laporan klasifikasi berdasarkan prediksi
        return self.score(y_test=self.y_test, y_pred=pred)

    def predict(self, X_test, output=None):
        """
        Melakukan prediksi menggunakan model yang telah diatur sebelumnya.

        Parameters:
            X_test (array-like or DataFrame): Data uji yang akan diprediksi.
            output (str, optional): Jika "dataframe", hasil prediksi akan dikembalikan dalam bentuk DataFrame. Defaultnya adalah None.

        Returns:
            array-like, DataFrame, or str: Hasil prediksi. Jika output adalah "dataframe", hasil prediksi akan dikembalikan dalam bentuk DataFrame.
                                            Jika model belum diatur sebelumnya, akan mengembalikan "None".

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Mengatur model dengan algoritma yang ditentukan
            model.set_model("RandomForestClassifier")

            # Melakukan prediksi menggunakan data uji
            predictions = model.predict(X_test)

            # Melakukan prediksi dan mengembalikan hasil dalam bentuk DataFrame
            predictions_df = model.predict(X_test, output="dataframe")
        """
        # Memeriksa apakah model telah diatur sebelumnya
        if not hasattr(self, "model"):
            return "None"

        # Melakukan normalisasi data uji menggunakan scaler yang telah disimpan
        X_test = self.scaler.transform(X_test)

        # Melakukan prediksi menggunakan model yang telah diatur
        pred = self.model[1].predict(X_test)

        # Jika output yang diminta adalah DataFrame
        if output == "dataframe":
            # Mengembalikan hasil prediksi dalam bentuk DataFrame
            X_test = self.scaler.inverse_transform(X_test)
            result = np.concatenate((X_test, pred[:, np.newaxis]), axis=1)
            column_names = [f"X{i}" for i in range(X_test.shape[1])]
            column_names.append("pred")
            return pd.DataFrame(result, columns=column_names)

        # Mengembalikan hasil prediksi
        return pred

    def save_model(self):
        """
        Menyimpan model ke dalam bentuk serialisasi menggunakan modul pickle.

        Returns:
            bytes: Objek model yang telah diserialisasi.

        Example:
            # Inisialisasi objek model
            model = MyModel()

            # Menyimpan model ke dalam bentuk serialisasi
            serialized_model = model.save_model()
        """
        # Melakukan serialisasi objek model menggunakan modul pickle
        return pickle.dumps(self)

    # Getter
    @property
    def get_X(self):
        return self.X

    @property
    def get_y(self):
        return self.y

    @property
    def get_X_train(self):
        return self.X_train

    @property
    def get_X_test(self):
        return self.X_test

    @property
    def get_params_from_model(self):
        if not self.model:
            return None
        return self.model[1].get_params()

    @property
    def get_result_compare_models(self):
        if not hasattr(self, "result_compare_models"):
            return None
        return self.result_compare_models

    @property
    def get_list_models(self):
        return list(self.ALGORITHM.keys())

    @property
    def get_params_required(self):
        if not hasattr(self, "model"):
            return "None"
        # return self.model[1]
        pass
