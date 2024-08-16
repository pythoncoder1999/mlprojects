#all common functions for the whole functionality
import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train, y_train, x_test, y_test, model, params):
    try:
        report = {}
        for i in range(len(list(model))):
            model1 = list(model.values())[i]
            para = params[list(model.keys())[i]]

            gs = GridSearchCV(model1,para,cv = 3, n_jobs=-1)
            gs.fit(x_train, y_train)
            
            model1.set_params(**gs.best_params_)
            model1.fit(x_train,y_train)

            y_train_pred = model1.predict(x_train)
            y_test_pred = model1.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(model.keys())[i]] = test_model_score

            return report
    except Exception as e:
        raise CustomException(e,sys)
