from sklearn.model_selection import GridSearchCV, KFold, train_test_split,StratifiedKFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,TargetEncoder
import pandas as pd
import pickle
from abc import ABC, abstractmethod


import warnings
warnings.filterwarnings('always')  


class Parent(ABC):
    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def compare_model(self): ...

    # @abstractmethod
    # def find_best_model(self): ...

    @abstractmethod
    def score(self, y_true, y_pred): ...

    def check_imbalance(dataset:pd.DataFrame, target_column):
        # Hitung distribusi kelas target
        class_distribution = dataset[target_column].value_counts(normalize=True)
        class_minority = class_distribution[class_distribution==class_distribution.min()]

        # Periksa apakah salah satu kelas memiliki persentase di bawah ambang batas
        imbalance_threshold = 0.02  # Anda dapat menyesuaikan ambang batas sesuai kebutuhan

        status = False

        if class_distribution.var() >= imbalance_threshold:
            imbalance_info = f"""
            The {class_minority.index.tolist()} class has an imbalance of {class_minority.values}
            \nConsider handling class imbalance. 
            """
            print(imbalance_info)
            status = True
        return status

    def handling_missing_values(self,data:pd.DataFrame,imputation=None,inplace=False) -> pd.DataFrame:
        miss_value = data.isna().sum().to_dict()
        miss_value = {column: value for column, value in miss_value.items() if value != 0}
        data = data.copy()
        if imputation is None:
            for i in miss_value:
                if data[i].dtype == 'object':
                    val = data[i].mode()[0]
                    data[i] = data[i].fillna(val)
                elif data[i].dtype in ['int64','int32','float64','float32']:
                    val = data[i].median()
                    data[i] = data[i].fillna(val)
                else:
                    raise ValueError("Not Defined") 
        elif imputation == "drop":
            data = data.dropna()
        else:
            for i in miss_value:
                data[i] = data.fillna(imputation[i]) 
        if inplace:
            self.data = data
            return self
        else:
            return data   

    def encoding(data:pd.DataFrame,features_cat:str,target:str):
        target_encoder = TargetEncoder().fit(data[features_cat],data[target])
        X = target_encoder.transform(data[features_cat])
        for i,v in enumerate(features_cat):
            data[v] = X[:,i]
        data = data.drop(columns=[target])
        return data,target_encoder

    def encoding_predict(encoder,X_test,features_cat):
        X = encoder.transform(X_test[features_cat])
        for i,v in enumerate(features_cat):
            X_test[v] =  X[:,1]
        return X_test
        
    def decoding(data:pd.DataFrame,data_categories):
        for i in data_categories.columns:
            data[i] = data_categories[i]
        return data
    
    def check_col_categories(self,data:pd.DataFrame):
        i = []
        for j in data.columns:
            if data[j].dtype == 'object':
                i.append(j)
        return i

    def split_data(self,train_size):
        stratify = self.y if self.model_type == 'Classification' else None
        return train_test_split(self.X,self.y,random_state=self.seed,train_size=train_size,stratify=stratify)

    def find_best_params(self,algo_name,param_grid,X_train=None,y_train=None,n_splits=5):
        if algo_name not in self.ALGORITHM.keys():
            raise ValueError("Algorithm not found in the list.")
        if X_train == None and y_train == None:
            X_train = self.X
            y_train = self.y
        alg = self.ALGORITHM[algo_name]
        kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=self.seed)
        clf = GridSearchCV(estimator=alg,param_grid=param_grid,cv=kfold,scoring='accuracy')
        clf.fit(X_train,y_train)
        return clf.best_params_,clf.best_score_


    # Fit Method
    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        features: list,
        norm=True,
        seed=42
    ):
        data = data.copy()
        self.data = data[features]
        self.features = features
        self.seed = seed

        # Missing Values Handler
        if sum(data.isna().sum().values) > 0 :
            miss_value = {column: value for column, value in data.isna().sum().items() if value != 0}
            raise ValueError(f"Missing Value in {miss_value}")

        # Check Imbalance
        if self.model_type == 'Classification':
            self.status_imbalance = Parent.check_imbalance(data,target) 
        
        # Encoding Columns
        if any(self.check_col_categories(data[features])):
            data_categories = data[features].select_dtypes('object')
            cols_categories = data_categories.columns.tolist()
            X,encoder = Parent.encoding(data=data,features_cat=cols_categories,target=target)
            self.X_encoder = encoder
            self.cols_encoded = cols_categories
            self.data_categories = data_categories
        X = data[features]

        y = data[target].values
        if data[target].dtype == object:
            labelencoder = LabelEncoder().fit(y)
            y = labelencoder.transform(y)
            self.y_encoder = labelencoder

        # Features Norm
        if norm:
            scaler = MinMaxScaler()
            scaler.fit(X.values)
            X = scaler.transform(X.values)
            self.scaler = scaler
        else:
            X = X.values

        self.X = X
        self.y = y
        self.norm = norm

        return self


    def set_model(self, algo_name,X_train=None,y_train=None):
        if isinstance(X_train,np.ndarray) and isinstance(X_train,np.ndarray):
            pass
        else:
            X_train = self.X
            y_train = self.y

        self.X_train = X_train
        self.y_train = y_train
        try:
            if algo_name in self.ALGORITHM:
                model_instance = self.ALGORITHM[algo_name]
                model = model_instance.fit(X_train, y_train)
                self.model = (algo_name, model)
                return self
            else:
                raise ValueError("Algorithm not found in the list.")
        except Exception as e:
            print("Error:", e)
            return self
    
    def GridSearch(self,parameters) -> GridSearchCV():
        self.not_found("model")
        return GridSearchCV(self.model[1],parameters).fit(self.X,self.y)

    def hyperparameters(self, params):
        if not hasattr(self, "model"):
            raise ValueError("Model algorithm not found")

        algo_name = self.model[0]

        model_instance = self.model[1]

        params_required = list(model_instance.get_params().keys())

        for key in params:
            if key not in params_required:
                raise ValueError(
                    "Parameter '{}' is not required for this model.".format(key)
                )

        model = self.model[1]
        model = model.set_params(**params)

        self.model = algo_name, model

        return self

    def predict(self, X_test: pd.DataFrame, output=None):
        if not hasattr(self, "model"):
            raise NotImplementedError("Model Not Define")
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "The number of features in the test data is not equal to the number of features in the training data."
            ) 

        if type(X_test).__name__ not in ['ndarray','DataFrame']:
            raise ValueError("X_test data type must be ndarray or dataframe")

        X = X_test
        if isinstance(X_test,pd.DataFrame):
            if hasattr(self,'cols_encoded'):
                X = Parent.encoding_predict(self.X_encoder,X,self.cols_encoded)
            X = X.values
        else:
            if X_test.dtype == 'object':
                X = pd.DataFrame(data=X,columns=self.features)
                X = Parent.encoding_predict(self.X_encoder,X,self.cols_encoded)
                X = X.values

        if hasattr(self,'scaler'):
            if (X.min() >= 0) and (X.max() <=1):
                ...
            else:
                X = self.scaler.transform(X)

        # Melakukan prediksi menggunakan model yang telah diatur
        pred = self.model[1].predict(X)
        
        if hasattr(self,'y_encoder'):
            pred = self.y_encoder.inverse_transform(pred)
        # Jika output yang diminta adalah DataFrame
        if output == "dataframe":
            X_test['pred'] = pred
            return X_test

        return pred

    def save_model(self):
        return pickle.dumps(self)

    def not_found(self, attr: str):
        if not hasattr(self, attr):
            raise ValueError(f"{attr} not found")

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

    # def model(self):
    #     return self.model

    @property
    def get_params_from_model(self):
        self.not_found("model")
        return self.model[1].get_params()

    @property
    def get_result_compare_models(self):
        self.not_found("result_compare_models")
        return self.result_compare_models

    @property
    def get_metrics(self):
        return self.METRICS

    @property 
    def get_algorithm(self):
        return self.ALGORITHM 

    @property
    def get_list_models(self):
        return list(self.ALGORITHM.keys())

    @property
    def get_algorithm_from_find_best_model(self):
        self.not_found("best_algorithm")
        return self.best_algorithm
