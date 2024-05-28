from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    SelectKBest,
    chi2,
    VarianceThreshold,
    RFE,
    f_classif,
)
from skfeature.function.similarity_based.fisher_score import fisher_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .dataset import Dataset

from . import Classification, Regression


# Feature Selection (Filter Methods)
def feature_select_information_gain(
    X: pd.DataFrame, y: pd.DataFrame, n_features: int, t: str
) -> list:
    """
    Selects the top `n_features` based on information gain for classification or regression tasks.

    This function evaluates the importance of features in a dataset using mutual information.
    It supports both classification and regression tasks, and selects the most important features
    according to the specified number of features to select.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe containing the feature columns.

    y : pd.Series
        The target variable column.

    n_features : int
        The number of top features to select based on information gain.

    t : str
        The type of task. It can be either "classification" or "regression".

    Returns
    -------
    list
        A list of the selected feature names ranked by their importance.

    Raises
    ------
    ValueError
        If `t` is not "classification" or "regression".

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> selected_features = feature_select_information_gain(X, y, n_features=2, t='classification')
    >>> print(selected_features)
    ['petal length (cm)', 'petal width (cm)']
    """
    features = X.columns.tolist()
    if t == "classification":
        importances = list(zip(features, mutual_info_classif(X, y)))
        importances.sort(key=lambda x: x[1], reverse=True)
    elif t == "regression":
        importances = list(zip(features, mutual_info_regression(X, y)))
        importances.sort(key=lambda x: x[1], reverse=True)
    else:
        raise ValueError("t is not found !")
    importances = [i[0] for i in importances[:n_features]]
    return importances


def feature_select_chi_square(
    X: pd.DataFrame, y: pd.DataFrame, n_features: int
) -> list:
    """
    Selects the top `n_features` based on the chi-square statistic for classification tasks.

    This function evaluates the importance of categorical features in a dataset using the chi-square
    statistical test. It selects the most important features according to the specified number of features to select.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe containing the feature columns.

    y : pd.Series
        The target variable column.

    n_features : int
        The number of top features to select based on the chi-square statistic.

    Returns
    -------
    list
        A list of the selected feature names ranked by their chi-square scores.

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> selected_features = feature_select_chi_square(X, y, n_features=2)
    >>> print(selected_features)
    ['petal length (cm)', 'petal width (cm)']
    """
    selector = SelectKBest(chi2, k=n_features)
    selector.fit(X, y)
    return selector.get_feature_names_out()


def feature_select_anova(X: pd.DataFrame, y: pd.DataFrame, n_features: int):
    """
    Perform feature selection using ANOVA F-test.

    This method selects the top `n_features` features from the input dataframe `X`
    based on the ANOVA F-test. It uses the `SelectKBest` class with the `f_classif`
    scoring function to rank features by their importance.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe containing feature values.

    y : pd.DataFrame
        The target values corresponding to the input features.

    n_features : int
        The number of top features to select.

    Returns
    -------
    list
        A list of the names of the selected features.

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.DataFrame(data.target, columns=["target"])
    >>> selected_features = feature_select_anova(X, y, n_features=2)
    >>> print(selected_features)
    ['petal length (cm)', 'petal width (cm)']
    """
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    return selector.get_feature_names_out()


def feature_select_fisher_score(
    X: pd.DataFrame, y: pd.DataFrame, n_features: int
) -> list:
    """
    Perform feature selection using Fisher Score.

    This method selects the top `n_features` from the input dataframe `X` based on the Fisher Score.
    Fisher Score is a supervised feature selection method that evaluates the importance of a feature
    by measuring the discriminative power of each feature with respect to the target `y`.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe containing feature values.

    y : pd.DataFrame
        The target values corresponding to the input features.

    n_features : int
        The number of top features to select.

    Returns
    -------
    list
        A list of the names of the selected features.

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>> from skfeature.function.similarity_based import fisher_score
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.DataFrame(data.target, columns=["target"])
    >>> selected_features = feature_select_fisher_score(X, y, n_features=2)
    >>> print(selected_features)
    ['petal width (cm)', 'petal length (cm)']
    """
    importances = list(zip(X.columns, fisher_score(X.values, y.values)))
    importances.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in importances[:n_features]]


def feature_select_variance_threshold(X: pd.DataFrame, threshold: int) -> list:
    """
    Perform feature selection using Variance Threshold.

    This method selects features from the input dataframe `X` based on a variance threshold.
    Features with a variance lower than the threshold will be removed, as they are less likely
    to be informative.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe containing feature values.

    threshold : float
        The variance threshold. Features with a variance lower than this value will be removed.

    Returns
    -------
    list
        A list of the names of the selected features that have a variance above the threshold.

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> selected_features = feature_select_variance_threshold(X, threshold=0.5)
    >>> print(selected_features)
    ['petal length (cm)', 'petal width (cm)']
    """
    importances = VarianceThreshold(threshold=threshold).fit(X)
    return importances.get_feature_names_out()


# Feature Selection (wrapper methods)
def feature_select_rfe(X: pd.DataFrame, y: pd.DataFrame, n_features: int, t: str):
    """
    Perform feature selection using Recursive Feature Elimination (RFE).

    This method selects the top `n_features` from the input dataframe `X` using Recursive Feature
    Elimination with a specified model type for either classification or regression tasks.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe containing feature values.

    y : pd.DataFrame
        The target values corresponding to the input features.

    n_features : int
        The number of top features to select.

    t : str
        The type of task: "classification" or "regression". Determines which model to use for RFE.
        - If "classification", uses `RandomForestClassifier`.
        - If "regression", uses `RandomForestRegressor`.

    Returns
    -------
    list
        A list of the names of the selected features.

    Raises
    ------
    ValueError
        If `t` is neither "classification" nor "regression".

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.DataFrame(data.target, columns=["target"])
    >>> selected_features = feature_select_rfe(X, y, n_features=2, t="classification")
    >>> print(selected_features)
    ['petal width (cm)', 'petal length (cm)']
    """
    if t == "classification":
        model = RandomForestClassifier(random_state=42)
    elif t == "regression":
        model = RandomForestRegressor(random_state=42)

    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)
    return rfe.get_feature_names_out()


def feature_selection(
    dataframe: pd.DataFrame,
    target: str,
    n_features: int,
    threshold: int = 0.0,
    features: list = None,
    show_score: bool = True,
) -> dict:
    """
    Perform feature selection using various methods and evaluate model performance.

    This function applies multiple feature selection methods to a given dataframe, fits a
    classification or regression model based on the target variable type, and evaluates the model
    performance for each set of selected features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing features and the target variable.

    target : str
        The name of the target variable column in the dataframe.

    n_features : int
        The number of top features to select using feature selection methods.

    threshold : int, optional
        The threshold for variance threshold feature selection. Default is 0.0.

    features : list, optional
        A list of feature names to consider for selection. If not provided, all columns
        except the target column are used.

    Returns
    -------
    dict
        A dictionary where keys are the names of feature selection methods and values are
        the sets of selected features.

    Example
    -------
    >>> df = pd.read_csv("data.csv")
    >>> selected_features = feature_selection(dataframe=df, target="target", n_features=10)
    >>> print(selected_features)
    {
        'all': ['feature1', 'feature2', ...],
        'f_mutual': ['feature1', 'feature3', ...],
        'f_anova': ['feature2', 'feature4', ...],
        ...
    }
    """
    df = dataframe.copy()
    if not features:
        features = df.drop(target, axis=1).columns.tolist()
        del df
    dataset = Dataset(dataframe=dataframe, norm=True)
    dataset.fit(features=features, target=target)
    X_train = dataset.train[features]
    y_train = dataset.train[target]
    methods = {
        "all": features,
        "f_mutual": feature_select_information_gain(
            X_train, y_train, n_features=n_features, t=dataset.t
        ),
        "f_anova": feature_select_anova(X_train, y_train, n_features=n_features),
        "f_chi_square": feature_select_chi_square(
            X_train, y_train, n_features=n_features
        ),
        "f_fisher": feature_select_fisher_score(
            X_train, y_train, n_features=n_features
        ),
        "f_variance": feature_select_variance_threshold(X_train, threshold=threshold),
        "f_rfe": feature_select_rfe(
            X_train, y_train, n_features=n_features, t=dataset.t
        ),
    }
    result = {}
    for m, f in methods.items():
        dataset.features = f
        if dataset.t == "classification":
            model = Classification(dataset=dataset)
        else:
            model = Regression(dataset=dataset)
        res = model.compare_model(output="only_score", train_val=True, verbose=False)
        result[m] = res

    if show_score:
        print(pd.DataFrame().from_dict(result, orient="index"))
    return methods


def handling_missing_values(
    dataframe: pd.DataFrame, imputation: dict = None, threshold: float = 0.3
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame by either imputing or dropping columns based on a threshold.

    This function handles missing values in the given DataFrame. It can perform custom imputation using the provided
    dictionary of imputers, drop columns with missing values exceeding a specified threshold, and impute the remaining
    missing values using median for numerical columns and most frequent value for categorical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing missing values.

    imputation : dict, optional
        A dictionary where keys are column names and values are imputer instances from sklearn.impute. If provided,
        these imputers will be used to fill missing values in the specified columns.

    threshold : float, default=0.3
        A float value between 0 and 1 that specifies the maximum allowable fraction of missing values in a column.
        Columns with a fraction of missing values greater than or equal to this threshold will be dropped.

    Returns
    -------
    pd.DataFrame
        A DataFrame with missing values handled according to the specified parameters.

    Raises
    ------
    ValueError
        If the threshold is greater than or equal to 1.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.impute import SimpleImputer
    >>> data = {'A': [1, 2, None, 4], 'B': [None, 2, 3, 4], 'C': [1, 2, 3, None]}
    >>> df = pd.DataFrame(data)
    >>> imputer = SimpleImputer(strategy='mean')
    >>> handled_df = handling_missing_values(df, imputation={'A': imputer}, threshold=0.5)
    col : B dropped! | miss_value : 1 | threshold : 2.0
    >>> print(handled_df)
         A    C
    0  1.0  1.0
    1  2.0  2.0
    2  2.333333  3.0
    3  4.0  2.0

    Notes
    -----
    - This function creates a copy of the input DataFrame to avoid modifying the original DataFrame.
    - Columns with missing values exceeding the specified threshold are dropped.
    - For remaining columns with missing values, numerical columns are imputed using the median, and categorical
      columns are imputed using the most frequent value.
    """
    if threshold >= 1:
        raise ValueError("threshold >= len(dataframe)")
    threshold = threshold * len(dataframe)
    dataframe = dataframe.copy()
    cols = dataframe.isna().sum().to_dict()

    if imputation:
        for col, imp in imputation.items():
            x = dataframe[col].values.reshape(-1, 1)
            x = imp.fit_transform(x)
            dataframe[col] = x.reshape(-1)
            del cols[col]

    # Identify columns to drop based on threshold
    cols_to_drop = [col for col, count in cols.items() if count >= threshold]
    for col in cols_to_drop:
        print(
            f"col : {col} dropped! | miss_value : {cols[col]} | threshold : {threshold}"
        )
        dataframe.drop(columns=col, inplace=True)
        del cols[col]

    # Remaining columns with missing values
    missing_col = [col for col in cols if cols[col] != 0]

    if missing_col:
        # Impute numerical columns
        numerical_cols = (
            dataframe[missing_col].select_dtypes(include=np.number).columns.tolist()
        )
        if numerical_cols:
            imp = SimpleImputer(strategy="median")
            x = dataframe[numerical_cols].values
            x = imp.fit_transform(x)
            dataframe[numerical_cols] = x

        # Impute categorical columns
        categorical_cols = (
            dataframe[missing_col].select_dtypes(include="object").columns.tolist()
        )
        if categorical_cols:
            imp = SimpleImputer(strategy="most_frequent")
            x = dataframe[categorical_cols].values
            x = imp.fit_transform(x)
            dataframe[categorical_cols] = x

    return dataframe


def sampling(
    dataframe: pd.DataFrame,
    features: list,
    target: str,
    method: str = "smote",
    sampling_strategy: str = "auto",
    seed: int = 42,
):
    """
    Applies sampling techniques to balance the target classes in the dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the features and target column.

    features : list
        A list of feature column names to be used for sampling.

    target : str
        The name of the target column to be balanced.

    method : str, optional
        The sampling method to be used. Options are:
        - "under" : Random under-sampling
        - "over" : Random over-sampling
        - "smote" : Synthetic Minority Over-sampling Technique (default)

    sampling_strategy : str, optional
        The sampling strategy to use. Default is "auto".

    seed : int, optional
        The random seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        The dataframe with the target classes balanced according to the specified method.

    Raises
    ------
    ValueError
        If an invalid sampling method is provided.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5, 6],
    ...     'feature2': [7, 8, 9, 10, 11, 12],
    ...     'target': [0, 0, 0, 1, 1, 1]
    ... })
    >>> sampled_df = sampling(df, features=['feature1', 'feature2'], target='target', method='over')
    >>> sampled_df['target'].value_counts()
    0    3
    1    3
    dtype: int64
    """
    if method == "under":
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=seed
        )
    elif method == "over":
        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy, random_state=seed
        )
    elif method == "smote":
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
    else:
        raise ValueError("method not found !")
    X, y = sampler.fit_resample(dataframe[features], dataframe[target])
    X[target] = y.values
    return X


def pca(dataframe: pd.DataFrame, features: list, n_components: int) -> np.ndarray:
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the input data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the features.

    features : list
        A list of feature column names to be used for PCA.

    n_components : int
        The number of principal components to retain.

    Returns
    -------
    np.ndarray
        The transformed data after PCA.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'feature1': [1, 2, 3],
    ...     'feature2': [4, 5, 6],
    ...     'feature3': [7, 8, 9]
    ... })
    >>> transformed_data = pca(df, features=['feature1', 'feature2', 'feature3'], n_components=2)
    >>> transformed_data.shape
    (3, 2)  # Assuming 3 samples and 2 principal components
    """
    pc = PCA(n_components=n_components)
    pc.fit(dataframe[features])
    X = pc.transform(dataframe[features])
    return X


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """
    Detect outliers in a specified column of a DataFrame using the Z-score method.

    This function calculates the Z-score for each value in the specified column and identifies
    outliers based on the provided threshold. The Z-score is a measure of how many standard deviations
    a data point is from the mean. Outliers are data points with Z-scores greater than the given threshold.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.

    column : str
        The name of the column in which to detect outliers.

    threshold : float, optional
        The Z-score threshold above which a data point is considered an outlier. Default is 3.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rows of the input DataFrame that are considered outliers based on the Z-score.

    Example
    -------
    >>> data = {'values': [10, 12, 12, 13, 12, 15, 14, 16, 100]}
    >>> df = pd.DataFrame(data)
    >>> outliers = detect_outliers_zscore(df, column='values')
    >>> print(outliers)
    values   z_score
    8     100  3.280537

    Notes
    -----
    - This function creates a copy of the input DataFrame to avoid modifying the original DataFrame.
    - A new column 'z_score' is added to the copied DataFrame, which contains the calculated Z-scores.
    - The original DataFrame remains unmodified.
    """

    df = df.copy()
    df["z_score"] = (df[column] - df[column].mean()) / df[column].std()
    outliers = df[np.abs(df["z_score"]) > threshold]
    del df
    return outliers


def detect_outliers_iqr(df: pd.DataFrame, column: str):
    """
    Detect outliers in a specified column of a DataFrame using the Interquartile Range (IQR) method.

    This function identifies outliers in a specified column of the DataFrame based on the Interquartile Range (IQR).
    Outliers are data points that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.

    column : str
        The name of the column in which to detect outliers.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rows of the input DataFrame that are considered outliers based on the IQR method.

    Example
    -------
    >>> data = {'values': [10, 12, 12, 13, 12, 15, 14, 16, 100]}
    >>> df = pd.DataFrame(data)
    >>> outliers = detect_outliers_iqr(df, column='values')
    >>> print(outliers)
    values
    8     100

    Notes
    -----
    - This function creates a copy of the input DataFrame to avoid modifying the original DataFrame.
    - Outliers are identified based on the lower and upper bounds calculated using the IQR method.
    - The original DataFrame remains unmodified.
    """
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    del df
    return outliers


__all__ = [
    "feature_selection",
    "feature_select_rfe",
    "feature_select_anova",
    "feature_select_chi_square",
    "feature_select_fisher_score",
    "feature_select_variance_threshold",
    "feature_select_information_gain",
    "handling_missing_values",
    "pca",
    "sampling",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
]
