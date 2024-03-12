from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import pandas as pd
import pickle
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from abc import ABC, abstractmethod


class Parent(ABC):
    @abstractmethod
    def __str__(self) -> str:
        ...
    
    
    @abstractmethod
    def compare_model(self):
        ...
        
    @abstractmethod
    def find_best_model(self):
        ...
    
    
    def encoding(data):
        encoders = {}
        for i in data.columns:
            if data[i].dtype == 'object':
                encoder = LabelEncoder().fit(data[i])
                data[i] = encoder.transform(data[i])
                encoders[i] = encoder
        return data,encoders
    
    def decoding(data,encoder):
        data = data.copy()
        for i in encoder:
            data[i] = encoder[i].inverse_transform(data[i])
        return data

    def encoding_predict(data,encoder):
        data = data.copy()
        for i in encoder:
            data[i] = encoder[i].transform(data[i])
        return data
        

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        features: list,
        norm=True,
        split_data=0.2,
        all_data_train=False,
    ):

        # Memasukkan data, fitur, dan target ke dalam objek model
        self.norm = norm
        self.data = data[features]

    
        X,data_encoder = Parent.encoding(data[features].copy())
        self.data_encoder = data_encoder
        
        self.X = X.values
        self.encoded_cols = list(self.data_encoder.keys())
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

        if self.model_type == "Classification":
            return classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
        elif self.model_type == "Regression":
            return {
                "MAE": mean_absolute_error(y_pred=y_pred, y_true=y_test),
                "MAPE": mean_absolute_percentage_error(y_pred=y_pred, y_true=y_test),
                "MSE": mean_squared_error(y_pred=y_pred, y_true=y_test),
            }
    def set_model(self, algo_name):

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

    def predict(self, X_test:pd.DataFrame, output=None):
        # Memeriksa apakah model telah diatur sebelumnya
        if not hasattr(self, "model"):
            raise NotImplementedError("Model Not Define")
        if X_test.shape[1] != self.X.shape[1]:
            raise ValueError("The number of features in the test data is not equal to the number of features in the training data.")

        X_test = Parent.encoding_predict(data=X_test,encoder=self.data_encoder)
        X_test = X_test.values

        # Melakukan normalisasi data uji menggunakan scaler yang telah disimpan
        if X_test is not self.X_test:
            if self.norm:
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
        # Melakukan serialisasi objek model menggunakan modul pickle
        return pickle.dumps(self)
    
    def not_found(self,attr:str):
        if not hasattr(self,attr):
            raise ValueError(f'{attr} not found')
        
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
        self.not_found('model')
        return self.model[1].get_params()

    @property
    def get_result_compare_models(self):
        self.not_found("result_compare_models")
        return self.result_compare_models

    @property
    def get_list_models(self):
        return list(self.ALGORITHM.keys())
    
    @property
    def get_algorithm_from_find_best_model(self):
        self.not_found("best_algorithm")
        return self.best_algorithm
