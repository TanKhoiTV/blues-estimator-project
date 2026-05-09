"""
Model Comparison Module.

This module provides the ModelComparison class to train, evaluate,
and compare multiple regression models.
"""

from typing import Dict, Any
import pandas as pd

class ModelComparison:
    """
    Trains and compares multiple regression models.
    
    Supports Linear Regression, Ridge Regression, Lasso Regression,
    and ElasticNet Regression.
    """
    
    def __init__(self) -> None:
        """
        Initialize the ModelComparison object.
        """
        pass
        
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a Linear Regression model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
            
        Returns
        -------
        Any
            The trained Linear Regression model instance.
        """
        pass
        
    def train_ridge_regression(self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Any:
        """
        Train a Ridge Regression model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        alpha : float, optional
            Regularization strength (default is 1.0).
            
        Returns
        -------
        Any
            The trained Ridge Regression model instance.
        """
        pass
        
    def train_lasso_regression(self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Any:
        """
        Train a Lasso Regression model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        alpha : float, optional
            Regularization strength (default is 1.0).
            
        Returns
        -------
        Any
            The trained Lasso Regression model instance.
        """
        pass

    def train_elasticnet_regression(self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0, l1_ratio: float = 0.5) -> Any:
        """
        Train an ElasticNet Regression model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        alpha : float, optional
            Constant that multiplies the penalty terms (default is 1.0).
        l1_ratio : float, optional
            The ElasticNet mixing parameter (default is 0.5).
            
        Returns
        -------
        Any
            The trained ElasticNet Regression model instance.
        """
        pass
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model on testing data.
        
        Parameters
        ----------
        model : Any
            The trained regression model.
        X_test : pd.DataFrame
            The testing feature data.
        y_test : pd.Series
            The testing target data.
            
        Returns
        -------
        dict of str to float
            A dictionary containing evaluation metrics (e.g., MSE, R2).
        """
        pass
        
    def compare_metrics(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Train and evaluate all supported models and compile their metrics.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        X_test : pd.DataFrame
            The testing feature data.
        y_test : pd.Series
            The testing target data.
            
        Returns
        -------
        pd.DataFrame
            A dataframe containing comparison metrics for all trained models.
        """
        pass
        
    def generate_summary(self, metrics_df: pd.DataFrame) -> str:
        """
        Generate a text summary of the model comparison results.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            The dataframe containing model comparison metrics.
            
        Returns
        -------
        str
            A formatted summary string of the results.
        """
        pass
