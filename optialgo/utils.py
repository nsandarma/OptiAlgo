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
from sklearn.ensemble import RandomForestClassifier


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
    importances = [i[0] for i in importances[: n_features + 1]]
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
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    return selector.get_feature_names_out()


def feature_select_fisher_score(
    X: pd.DataFrame, y: pd.DataFrame, n_features: int
) -> list:
    importances = list(zip(X.columns, fisher_score(X.values, y.values)))
    importances.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in importances[:n_features]]


def feature_select_variance_threshold(X: pd.DataFrame, threshold: int) -> list:
    importances = VarianceThreshold(threshold=threshold).fit(X)
    return importances.get_feature_names_out()


def feature_select_rfe(X: pd.DataFrame, y: pd.DataFrame, n_features):
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)
    return rfe.get_feature_names_out()


def handling_missing_values(
    dataframe: pd.DataFrame, imputation: dict = None, lim: float = 0.3
) -> pd.DataFrame:
    """
    Handles missing values in the dataframe by either imputing them or dropping columns
    with excessive missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the data with potential missing values.

    imputation : dict, optional
        A dictionary specifying custom imputers for specific columns.
        If provided, these imputers will be used to fill missing values.

    lim : float, optional
        The threshold for dropping columns with missing values.
        Columns with a proportion of missing values greater than or equal to `lim`
        will be dropped. Default is 0.3 (30%).

    Returns
    -------
    pd.DataFrame
        The dataframe with missing values handled either through imputation or column removal.

    Raises
    ------
    ValueError
        If `lim` is greater than or equal to 1, as this would imply dropping all rows.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan],
    ...     'B': [4, np.nan, np.nan],
    ...     'C': ['a', 'b', np.nan]
    ... })
    >>> imputation = {'A': SimpleImputer(strategy='mean')}
    >>> cleaned_df = handling_missing_values(df, imputation=imputation, lim=0.5)
    col : B dropped ! | miss_value : 2 | threshold : 1.5
    >>> cleaned_df
         A    C
    0  1.0    a
    1  2.0    b
    2  1.5  NaN  # Assuming mean imputation for column 'A'
    """
    if lim >= 1:
        raise ValueError("lim >= len(dataframe)")
    lim = lim * len(dataframe)

    miss_value = dataframe.isna().sum().to_dict()
    col_missing = [col for col in miss_value if miss_value[col] != 0]

    if imputation:
        for col, imp in imputation.items():
            x = dataframe[col].values.reshape(-1, 1)
            x = imp.fit_transform(x)
            dataframe[col] = x.reshape(-1)
        return dataframe
    else:
        for i, col in enumerate(col_missing):
            if miss_value[col] >= lim:
                print(
                    "col : {} dropped ! | miss_value : {} | threshold : {}".format(
                        col, miss_value[col], lim
                    )
                )
                del dataframe[col], col_missing[i]

        numerical_cols = (
            dataframe[col_missing].select_dtypes(include=[np.number]).columns.tolist()
        )
        if numerical_cols:
            imp = SimpleImputer(strategy="median")
            for col in numerical_cols:
                x = dataframe[col].values.reshape(-1, 1)
                x = imp.fit_transform(x)
                dataframe[col] = x.reshape(-1)

        categorical_cols = (
            dataframe[col_missing].select_dtypes("object").columns.tolist()
        )
        if categorical_cols:
            imp = SimpleImputer(strategy="most_frequent")
            for col in categorical_cols:
                x = dataframe[col].values.reshape(-1, 1)
                x = imp.fit_transform(x)
                dataframe[col] = x.reshape(-1)

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
