from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px

from utils.transform_data import zScoreRegularization

alphas = np.logspace(-3, 2, 100)

def initialize(train_x, train_y,test_x,test_y):
    train_x = zScoreRegularization(train_x.select_dtypes(include=["float64"]))
    test_x = zScoreRegularization(test_x.select_dtypes(include=["float64"]))
    train_y = zScoreRegularization(train_y)
    test_y = zScoreRegularization(test_y)


def ols(train_x, train_y,test_x,test_y):
    #initialize(train_x, train_y,test_x,test_y)
    slrRegressor = LinearRegression().fit(train_x, train_y)
    y_pred_ols = slrRegressor.predict(test_x)
    mse_ols = mean_squared_error(test_y, y_pred_ols)
    r2_ols = r2_score(test_y, y_pred_ols)
    return slrRegressor,mse_ols,r2_ols

def ridgeCV(train_x, train_y,test_x,test_y):
    #initialize(train_x, train_y,test_x,test_y)
    ridge_cv_model = RidgeCV(alphas=alphas, cv=5)
    ridge_cv_model.fit(train_x, train_y)
    y_pred_ridge_cv = ridge_cv_model.predict(test_x)
    # Evaluate the model
    mse_ridge_cv = mean_squared_error(test_y, y_pred_ridge_cv)
    r2_ridge_cv = r2_score(test_y, y_pred_ridge_cv)

    return ridge_cv_model,mse_ridge_cv,r2_ridge_cv

def lassoCV(train_x, train_y,test_x,test_y):
    #initialize(train_x, train_y,test_x,test_y)
    lasso_cv_model = LassoCV(alphas=alphas, cv=5)
    lasso_cv_model.fit(train_x, train_y)
    y_pred_lasso_cv = lasso_cv_model.predict(test_x)
    # Evaluate the model
    mse_lasso_cv = mean_squared_error(test_y, y_pred_lasso_cv)
    r2_lasso_cv = r2_score(test_y, y_pred_lasso_cv)

    return lasso_cv_model,mse_lasso_cv,r2_lasso_cv

def generate_graph_coeff(train_x,lasso_cv_model,ridge_cv_model):
    feature_names = train_x.columns
    coefLasso = lasso_cv_model.coef_
    coefRidge = ridge_cv_model.coef_

    fig = px.bar(x=feature_names,y=coefLasso)
    fig.show()

    fig = px.bar(x=feature_names,y=coefRidge)
    fig.show()