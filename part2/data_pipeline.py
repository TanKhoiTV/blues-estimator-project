"""
Data Pipeline Module.

This module provides the DataPipeline class for loading, cleaning,
and preprocessing datasets for machine learning models.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataPipeline:
    """
    Orchestrate the data preprocessing pipeline.

    This class handles loading data, missing value imputation,
    duplicate removal, categorical encoding, feature scaling,
    and train-test splitting.
    """

    def __init__(self) -> None:
        """Initialize the DataPipeline object."""
        # Store statistics learned from training data
        self.numeric_means_ = {}
        self.numeric_stds_ = {}

        # Store categorical mappings
        self.categorical_values_ = {}
        self.categorical_modes_ = {}

        # Store column types
        self.numeric_columns_ = []
        self.categorical_columns_ = []

        # Track encoded columns from training data
        self.encoded_columns_ = []

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from a specified file path.

        Parameters
        ----------
        file_path : str
            Path to the CSV dataset file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        return pd.read_csv(file_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with duplicate rows removed.
        """
        return df.drop_duplicates().reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y=None) -> None:
        """
        Learn preprocessing parameters from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature dataset.

        y : optional
            Target values. Unused in this pipeline.

        Returns
        -------
        None
        """
        X = X.copy()

        # Detect numerical and categorical columns
        self.numeric_columns_ = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        self.categorical_columns_ = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        # Learn statistics for numerical columns
        for col in self.numeric_columns_:
            self.numeric_means_[col] = X[col].mean()
            self.numeric_stds_[col] = X[col].std()

            # Prevent division by zero
            if self.numeric_stds_[col] == 0:
                self.numeric_stds_[col] = 1

        # Learn categorical mappings
        for col in self.categorical_columns_:
            self.categorical_values_[col] = X[col].dropna().unique().tolist()
            mode = X[col].mode(dropna=True)
            self.categorical_modes_[col] = None if mode.empty else mode[0]

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using stored training statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with imputed missing values.
        """
        df = df.copy()

        # Numerical columns -> mean imputation
        for col in self.numeric_columns_:
            df[col] = df[col].fillna(self.numeric_means_[col])

        # Categorical columns -> mode imputation
        for col in self.categorical_columns_:
            df[col] = df[col].fillna(self.categorical_modes_[col])

        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Encoded dataset.
        """
        df = df.copy()

        # Ensure categories are consistent with training data
        for col in self.categorical_columns_:
            df[col] = df[col].where(
                df[col].isin(self.categorical_values_[col]), other=None
            )

            df[col] = pd.Categorical(df[col], categories=self.categorical_values_[col])

        df = pd.get_dummies(df, columns=self.categorical_columns_)

        # Store encoded columns during training
        if not self.encoded_columns_:
            self.encoded_columns_ = df.columns.tolist()

        # Add missing columns in test data
        for col in self.encoded_columns_:
            if col not in df.columns:
                df[col] = 0

        # Remove unseen extra columns
        df = df[self.encoded_columns_]

        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using Z-score scaling.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Scaled dataset.
        """
        df = df.copy()

        for col in self.numeric_columns_:

            if col in df.columns:
                df[col] = (df[col] - self.numeric_means_[col]) / self.numeric_stds_[col]

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing using stored training parameters.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Transformed dataset.
        """
        X = X.copy()

        X = self.handle_missing_values(X)

        X = self.encode_categorical(X)

        X = self.scale_features(X)

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit and transform training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature dataset.

        y : optional
            Target values. Unused in this pipeline.

        Returns
        -------
        pd.DataFrame
            Transformed training dataset.
        """
        self.fit(X, y)

        return self.transform(X)

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into training and testing sets.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        target_column : str
            Name of the target column.

        test_size : float, default=0.2
            Fraction of dataset used for testing.

        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        tuple
            Training features, testing features,
            training labels, and testing labels.
        """
        X = df.drop(columns=[target_column])

        y = df[target_column]

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def preprocess(
        self, file_path: str, target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute the full preprocessing pipeline.

        Parameters
        ----------
        file_path : str
            Path to the dataset file.

        target_column : str
            Name of the target column.

        Returns
        -------
        tuple
            Preprocessed training features,
            testing features,
            training labels,
            and testing labels.
        """
        # Load dataset
        df = self.load_dataset(file_path)

        # Remove duplicates
        df = self.remove_duplicates(df)

        # Split BEFORE fitting to avoid leakage
        X_train, X_test, y_train, y_test = self.split_data(df, target_column)

        # Learn parameters ONLY from training data
        self.fit(X_train)

        # Transform train and test
        X_train = self.transform(X_train)

        X_test = self.transform(X_test)

        return X_train, X_test, y_train, y_test
