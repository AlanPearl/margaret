#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-

"""This module defines the Regressors class which contains several
   machine learning methods like RandomForest and K nearest neighbors.
   It can also calculate scores such as mean squared error, median absolute
   error, R^2 and training time.
"""

import time
from astropy.table import Table
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


class Regressors:
    """
    This class includs several machine learning regression methods which are
    random forest, k nearest neighbors, gradient boosting and xgboost
    to train and predict data.

    Example:

    # data_X and data_Y are training input and target data.
    reg = Regressor(data_X, data_Y)
    data_predict, data_test = reg.RFregressor(**kwargs)
    data_predict, data_test = reg.KNNregressor(**kwargs)
    data_predict, data_test = reg.KNN_dregressor(**kwargs)
    data_predict, data_test = reg.GBregressor(**kwargs)
    data_predict, data_test = reg.Xgboostregressor(**kwargs)

    # or use cross validation
    data_predict, data_test = reg.RFregressor(cross_validation=True)
    data_predict, data_test = reg.KNNregressor(cross_validation=True)
    data_predict, data_test = reg.KNN_dregressor(cross_validation=True)
    data_predict, data_test = reg.GBregressor(cross_validation=True)
    data_predict, data_test = reg.Xgboostregressor(cross_validation=True)

    """

    def __init__(self, data_X, data_Y, train_size=0.5, test_size=0.5, 
                 random_state=None, apply_pca=False):
        """
        Args:
            data_X: array-like or matrix of shape=[n_samples, n_features]
                The training input samples.
            data_Y: array-like, shape = [n_samples] or [n_samples, n_outputs]
                The target values.
            train_size: float
                Train test split ratio (if not using cross_validation)
            test_size: float
                Train test split ratio (if not using cross_validation)
            random_state: int
                Random seed from which to split the test and training sample
            apply_pca: bool
                If true, the training features will first be decomposed by PCA
                before they are used to train the regressor. PCA uses no
                information from the testing features
        """
        self._X = data_X
        self._Y = data_Y

        # Rescaling data
        scaler = StandardScaler()
        scaler.fit(self._X)
        self._scaled_X = scaler.transform(self._X)

        # Train-test split
        split = train_test_split(self._X, self._Y, self._scaled_X,
                                 train_size=train_size, test_size=test_size,
                                 random_state=random_state)
        
        # Apply principal component analysis
        if apply_pca:
            ndim = self._X.shape[1]
            self._X = PCA(n_components=ndim).fit_transform(self._X)
            self._scaled_X = PCA(n_components=ndim).fit_transform(self._scaled_X)
            
            pca = PCA(n_components=ndim)
            self._Xtrain = pca.fit_transform(split[0])
            self._Xtest = pca.transform(split[1])
            
            pca = PCA(n_components=ndim)
            self._scaled_Xtrain = pca.fit_transform(split[4])
            self._scaled_Xtest = pca.transform(split[5])
        else:
            self._Xtrain = split[0]
            self._Xtest = split[1]
            self._scaled_Xtrain = split[4]
            self._scaled_Xtest = split[5]
        
        self._Ytrain = split[2]
        self._Ytest = split[3]


    @staticmethod
    def Reg_scoring(Y_true, Y_pred):
        """
        Evaluate metrics on predicted data.

        Args:
            Y_true: array-like
                True values
            Y_pred: array-like
                Predictions

        Returns:
            score: dict
                A dictionary containing scores on mean squared error,
                median absolute error and R^2.

        """
        score = {}
        score['MSE'] = mean_squared_error(Y_true, Y_pred)
        score['MAE'] = median_absolute_error(Y_true, Y_pred)
        score['R2'] = r2_score(Y_true, Y_pred)

        return score

    @staticmethod
    def CV_scoring(reg, data_X, data_Y, n_folds=5):
        """
        Evaluate metrics on cross-validation predictions.

        Args:
            reg: estimator object implementing 'fit'
                The object to use to fit the data.
            data_X: array-like
                The training input samples.
            data_Y: array-like
                The target values.
            n_folds: int (default=5)
                The n_folds number for cross-validation

        Returns:
            score: dict
                A dictionary containing scores on mean squared error,
                median absolute error and R^2.

        """
        mtr = ['neg_mean_squared_error',
               'neg_median_absolute_error',
               'r2']

        score = {}
        sdict = cross_validate(reg, data_X, data_Y, cv=n_folds,
                               scoring=mtr)
        score['MSE'] = -sdict['test_neg_mean_squared_error'].mean()
        score['MAE'] = -sdict['test_neg_median_absolute_error'].mean()
        score['R2'] = sdict['test_r2'].mean()
        score['fit_time'] = sdict['fit_time'].mean()

        return score

    def RFregressor(self, cross_validation=False, n_folds=5, scoring=False,
                    n_estimators=50, max_depth=18, max_features='auto',
                    **kwargs):
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
            max_depth: int (default=18)
                The maximum depth of the tree.
                (Refer to RandomForestRegressor docstring)
            max_features: int, float, string or None (default='auto')
                The number of features to consider when looking for the
                best split
                (Refer to RandomForestRegressor docstring)
            scoring: boolean (default=False)
                Whether to evaluate the prediction. If true, evaluate
                mean squared error, median absolute error and R^2

        Returns:
            result: astropy table
                A table contains predictions and testing data. Also contains
                scores such as mean squared error, median absolute error, R^2
                and training time if scoring option is True.

        """
        reg = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                    **kwargs)

        if cross_validation:
            # Run cross-validation
            Y_predict = cross_val_predict(reg, self._X, self._Y,
                                          cv=n_folds)
            Y_test = self._Y

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.CV_scoring(reg, self._X, self._Y, n_folds=n_folds)

        else:
            # Train the regressor
            start = time.clock()
            reg.fit(self._Xtrain, self._Ytrain)
            end = time.clock()
            # Get predictions from regressor
            Y_predict = reg.predict(self._Xtest)
            Y_test = self._Ytest

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.Reg_scoring(Y_test, Y_predict)
                score['fit_time'] = end - start

        result = Table([Y_predict, Y_test],
                       names=['predict', 'test'])
        if scoring:
            for key in score:
                result.meta[key] = score[key]
        result.meta['model'] = reg

        return result

    def Xgboostregressor(self, cross_validation=False, n_folds=5,
                         scoring=False, booster='dart', learning_rate=0.1,
                         max_depth=5, objective='reg:linear',
                         normalize_type='tree', **kwargs):
        """
        Apply xgboost regression to train and predict data.

        Args:
            cross_validation: boolean (default=False)
                Whether to use cross-validation
            n_folds: int (default=5)
                The n_folds number if using cross-validation
            scoring: boolean (default=False)
                Whether to evaluate the prediction. If true, evaluate
                mean squared error, median absolute error and R^2
            booster: str (default='dart')
                Which booster to use. Can be 'gbtree', 'gblinear' or 'dart'.
            learning_rate: float (default0=0.1) range: [0,1]
                Step size shrinkage used in update to prevents overfitting.
            max_depth: int (default=5)
                Maximum depth of a tree.
            objective: str (default='reg:linear')
                Can be 'reg:linear', 'reg:logistic', 'reg:gamma' etc.
            normalize_type: str (default='tree')
                Type of normalization algorithm. Can be 'tree' or 'forest'.
            **kwarg: other kwargs for XGBRegressor()

        Returns:
            result: astropy table
                A table contains predictions and testing data. Also contains
                scores such as mean squared error, median absolute error, R^2
                and training time in meta data if scoring option is True.

        """
        reg = XGBRegressor(booster=booster, learning_rate=learning_rate,
                           max_depth=max_depth, objective=objective,
                           normalize_type=normalize_type, **kwargs)

        if cross_validation:
            # Run cross-validation
            Y_predict = cross_val_predict(reg, self._X, self._Y,
                                          cv=n_folds)
            Y_test = self._Y

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.CV_scoring(reg, self._X, self._Y, n_folds=n_folds)

        else:
            # Train the regressor
            start = time.clock()
            reg.fit(self._Xtrain, self._Ytrain)
            end = time.clock()
            # Get predictions from regressor
            Y_predict = reg.predict(self._Xtest)
            Y_test = self._Ytest

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.Reg_scoring(Y_test, Y_predict)
                score['fit_time'] = end - start

        result = Table([Y_predict, Y_test],
                       names=['predict', 'test'])
        if scoring:
            for key in score:
                result.meta[key] = score[key]
        result.meta['model'] = reg

        return result

    def GBregressor(self, cross_validation=False, n_folds=5,
                    scoring=False, max_depth=5, n_estimators=100,
                    learning_rate=0.1, **kwargs):
        """
        Apply gradient boosting regression to train and predict data.

        Args:
            cross_validation: boolean (default=False)
                Whether to use cross-validation
            n_folds: int (default=5)
                The n_folds number if using cross-validation
            scoring: boolean (default=False)
                Whether to evaluate the prediction. If true, evaluate
                mean squared error, median absolute error and R^2
            max_depth: int (default=5)
                Maximum depth of a tree.
            n_estimators: int (default=100)
                The number of boosting stages to perform.
            learning_rate: float (default0=0.1) range: [0,1]
                Step size shrinkage used in update to prevents overfitting.
            **kwarg: other kwargs for GradientBoostingRegressor().

        Returns
            result: astropy table
                A table contains predictions and testing data. Also contains
                scores such as mean squared error, median absolute error, R^2
                and training time in meta data if scoring option is True.

        """
        reg = GradientBoostingRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        **kwargs)

        if cross_validation:
            # Run cross-validation
            Y_predict = cross_val_predict(reg, self._X, self._Y,
                                          cv=n_folds)
            Y_test = self._Y

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.CV_scoring(reg, self._X, self._Y, n_folds=n_folds)

        else:
            # Train the regressor
            start = time.clock()
            reg.fit(self._Xtrain, self._Ytrain)
            end = time.clock()
            # Get predictions from regressor
            Y_predict = reg.predict(self._Xtest)
            Y_test = self._Ytest

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.Reg_scoring(Y_test, Y_predict)
                score['fit_time'] = end - start

        result = Table([Y_predict, Y_test],
                       names=['predict', 'test'])
        if scoring:
            for key in score:
                result.meta[key] = score[key]
        result.meta['model'] = reg

        return result

    def KNNregressor(self, cross_validation=False, n_folds=5,
                     n_neighbors=10, weights='uniform', scoring=False,
                     **kwargs):
        """
        Apply k nearest neighbor regression to train and predict data

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
            scoring: boolean (default=False)
                Whether to evaluate the prediction. If true, evaluate
                mean squared error, median absolute error and R^2

        Returns:
            result: astropy table
                A table contains predictions and testing data. Also contains
                scores such as mean squared error, median absolute error, R^2
                and training time if scoring option is True.

        """
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights,
                                  **kwargs)

        if cross_validation:
            # Run cross-validation
            if weights == 'uniform':
                Y_predict = cross_val_predict(reg, self._X, self._Y,
                                              cv=n_folds)
                Y_test = self._Y
                # Evaluate the prediction if scoring==True
                if scoring:
                    score = self.CV_scoring(reg, self._X, self._Y,
                                            n_folds=n_folds)

            if weights == 'distance':
                Y_predict = cross_val_predict(reg, self._scaled_X, self._Y,
                                              cv=n_folds)
                Y_test = self._Y
                # Evaluate the prediction if scoring==True
                if scoring:
                    score = self.CV_scoring(reg, self._scaled_X, self._Y,
                                            n_folds=n_folds)

        else:
            # Train the regressor
            if weights == 'uniform':
                start = time.clock()
                reg.fit(self._Xtrain, self._Ytrain)
                end = time.clock()
                # Get predictions from regressor
                Y_predict = reg.predict(self._Xtest)
                Y_test = self._Ytest
                # Evaluate the prediction if scoring==True
                if scoring:
                    score = self.Reg_scoring(Y_test, Y_predict)
                    score['fit_time'] = end - start

            if weights == 'distance':
                # Train the regressor
                start = time.clock()
                reg.fit(self._scaled_Xtrain, self._Ytrain)
                end = time.clock()
                # Get predictions from regressor
                Y_predict = reg.predict(self._scaled_Xtest)
                Y_test = self._Ytest
                # Evaluate the prediction if scoring==True
                if scoring:
                    score = self.Reg_scoring(Y_test, Y_predict)
                    score['fit_time'] = end - start

        result = Table([Y_predict, Y_test],
                       names=['predict', 'test'])
        if scoring:
            for key in score:
                result.meta[key] = score[key]
        result.meta['model'] = reg

        return result

    def KNN_dregressor(self, cross_validation=False, n_folds=5, scoring=False,
                       n_neighbors=10, weights='distance', **kwargs):
        """
        (This method is equivalent to KNNregressor when specifying
        weights='distance'.)

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
            scoring: boolean (default=False)
                Whether to evaluate the prediction. If true, evaluate
                mean squared error, median absolute error and R^2

        Returns:
            result: astropy table
                A table contains predictions and testing data. Also contains
                scores on mean squared error, median absolute error, R^2
                and training time if scoring option is True.

        """
        reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

        if cross_validation:
            # Run cross-validation
            Y_predict = cross_val_predict(reg, self._scaled_X, self._Y,
                                          cv=n_folds)
            Y_test = self._Y

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.CV_scoring(reg, self._scaled_X, self._Y,
                                        n_folds=n_folds)

        else:
            # Train the regressor
            start = time.clock()
            reg.fit(self._scaled_Xtrain, self._Ytrain)
            end = time.clock()
            # Get predictions from regressor
            Y_predict = reg.predict(self._scaled_Xtest)
            Y_test = self._Ytest

            # Evaluate the prediction if scoring==True
            if scoring:
                score = self.Reg_scoring(Y_test, Y_predict)
                score['fit_time'] = end - start

        result = Table([Y_predict, Y_test],
                       names=['predict', 'test'])
        if scoring:
            for key in score:
                result.meta[key] = score[key]
        result.meta['model'] = reg

        return result
