from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

import plotly.express as px


def random_forest(train_x, train_y,test_x,test_y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # Fit
    rf.fit(train_x, train_y)
    # Predict
    y_pred = rf.predict(test_x)
    # Evaluate
    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    return rf,mse,r2

def random_forest_importance_graph(data,rf):
    feature_names = data.columns
    coefRF = rf.feature_importances_

    fig = px.bar(x=feature_names,y=coefRF)
    fig.show()

def xgboost(train_x, train_y,test_x,test_y):
    xgb = XGBRegressor(random_state=42, n_estimators=100)
    xgb.fit(train_x, train_y)
    y_pred = xgb.predict(test_x)
    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    return xgb,mse,r2

def best_fit_xgboost_parameter(train_x, train_y):
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
    }
    xgb = XGBRegressor()
    search = RandomizedSearchCV(xgb, param_grid, cv=5, n_iter=50, scoring='mean_squared_error', verbose=1)
    search.fit(train_x, train_y)