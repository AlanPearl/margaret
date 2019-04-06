#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-

"""This module defines the Regressors class which contains several
   machine learning methods like RandomForest and K nearest neighbors.
"""


from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


class Regressors:
    """
    This class includs several machine learning regression methods which are
    random forest, k nearest neighbors (both uniform and distance weighted)
    to train and predict data.

    Example:

    # data_X and data_Y are training input and target data.
    reg = Regressor(data_X, data_Y)
    data_predict, data_test = reg.RFregressor()
    data_predict, data_test = reg.KNNregressor()
    data_predict, data_test = reg.KNN_dregressor()

    # or use cross validation
    data_predict, data_test = reg.RFregressor(cross_validation=True)
    data_predict, data_test = reg.KNNregressor(cross_validation=True)
    data_predict, data_test = reg.KNN_dregressor(cross_validation=True)

    """

    def __init__(self, data_X, data_Y, train_size=0.5, test_size=0.5):
        """
        Args:
            data_X: array-like or sparse matrix of shape = [n_samples, n_features]
                The training input samples.
            data_Y: array-like, shape = [n_samples] or [n_samples, n_outputs]
                The target values.
            train_size: float
                Train test split ratio (if not using cross_validation)
            test_size: float
                Train test split ratio (if not using cross_validation)

        """
        self._X = data_X
        self._Y = data_Y

        # Rescaling data
        scaler = StandardScaler()
        scaler.fit(self._X)
        self._scaled_X = scaler.transform(self._X)

        # Train-test split
        split = train_test_split(self._X, self._Y, self._scaled_X,
                                 train_size=train_size, test_size=test_size)
        self._Xtrain = split[0]
        self._Xtest = split[1]
        self._Ytrain = split[2]
        self._Ytest = split[3]
        self._scaled_Xtrain = split[4]
        self._scaled_Xtest = split[5]

    def RFregressor(self, cross_validation=False, n_folds=5,
                    n_estimators=50, max_depth=30, max_features='auto'):
        """
        Apply random forest regressor to train and predict data

        Args:
            cross_validation: boolean (default=False)
                Whether to use cross-validation
            n_folds: int (default=5)
                The n_folds number if using cross-validation
            n_estimators: int (default=50)
                The number of trees in the forest.
                (Refer to RandomForestRegressor docstring)
            max_depth: int (default=30)
                The maximum depth of the tree.
                (Refer to RandomForestRegressor docstring)
            max_features: int, float, string or None (default='auto')
                The number of features to consider when looking for the best split
                (Refer to RandomForestRegressor docstring)

        Returns:
            Y_predict: array-like
                The resulting predicted data
            Y_test: array-like
                Data for testing

        """
        reg = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    max_features=max_features)

        if cross_validation:
            Y_predict = cross_val_predict(reg, self._X, self._Y,
                                          cv=n_folds)
            Y_test = self._Y
            return Y_predict, Y_test

        else:
            reg.fit(self._Xtrain, self._Ytrain)
            Y_predict = reg.predict(self._Xtest)
            Y_test = self._Ytest
            return Y_predict, Y_test

    def KNNregressor(self, cross_validation=False, n_folds=5,
                     n_neighbors=10, weights='uniform'):
        """
        Apply k nearest neighbor regressor to train and predict data

        Args:
            cross_validation: boolean (default=False)
                Whether to use cross-validation
            n_folds: int (default=5)
                The n_folds number if using cross-validation
            n_neighbors: int (default=10)
                Number of neighbors to use.
                (Refer to KNeighborsRegressor docstring)
            weights: str (default='uniform')
                Weight function used in prediction.
                (Refer to KNeighborsRegressor docstring)

        Returns:
            Y_predict: array-like
                The resulting predicted data
            Y_test: array-like
                Data for testing

        """
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

        if cross_validation:
            Y_predict = cross_val_predict(reg, self._X, self._Y,
                                          cv=n_folds)
            Y_test = self._Y
            return Y_predict, Y_test

        else:
            reg.fit(self._Xtrain, self._Ytrain)
            Y_predict = reg.predict(self._Xtest)
            Y_test = self._Ytest
            return Y_predict, Y_test

    def KNN_dregressor(self, cross_validation=False, n_folds=5,
                       n_neighbors=10, weights='distance'):
        """
        Apply k nearest neighbor (weighted) regressor to train and predict data

        Args:
            cross_validation: boolean (default=False)
                Whether to use cross-validation
            n_folds: int (default=5)
                The n_folds number if using cross-validation
            n_neighbors: int (default=10)
                Number of neighbors to use.
                (Refer to KNeighborsRegressor docstring)
            weights: str (default='distance')
                Weight function used in prediction.
                (Refer to KNeighborsRegressor docstring)

        Returns:
            Y_predict: array-like
                The resulting predicted data
            Y_test: array-like
                Data for testing

        """
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

        if cross_validation:
            scaler = StandardScaler()
            scaler.fit(self._X)
            self._scaled_X = scaler.transform(self._X)
            Y_predict = cross_val_predict(reg, self._scaled_X, self._Y,
                                          cv=n_folds)
            Y_test = self._Y
            return Y_predict, Y_test

        else:
            scaler = StandardScaler()
            scaler.fit(self._Xtrain)
            self._scaled_Xtrain = scaler.transform(self._Xtrain)
            self._scaled_Xtest = scaler.transform(self._Xtest)
            reg.fit(self._scaled_Xtrain, self._Ytrain)
            Y_predict = reg.predict(self._scaled_Xtest)
            Y_test = self._Ytest
            return Y_predict, Y_test
