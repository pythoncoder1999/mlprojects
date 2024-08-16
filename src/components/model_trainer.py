import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,training_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                training_array[:,:-1],
                training_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Classifier":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree":{'criterion':['squared error','friedman_mse','absolute_error','poisson']},
                "Random Forest":{'n_estimators':[8,16,32,64,128,256]},
                "Gradient Boosting":{'learning_rate':[0.1,0.01,0.05,0.001],'subsample':[0.6,0.7,0.75,0.8,0.85,0.9], 'n_estimators':[8,16,32,64,128,256]},
                "Linear Regression":{},
                "K-Neighbors Classifier":{'n_neighbors':[5,7,9,11]},
                "XGBRegressor":{'learning_rate':[0.1,0.01,0.5,0.001], 'n_estimators':[8,16,32,64,128,256]},
                "AdaBoost Regressor":{'learning_rate':[0.1,0.01,0.05,0.001], 'n_estimators':[8,16,32,64,128,256]}
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train, x_test = x_test, y_test=y_test,model=models, params = params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if(best_model_score < 0.6):
                raise CustomException("No best model found",sys)
            logging.info("Best model found for training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model)

            return r2_score(y_test, best_model.predict(x_test))

        except Exception as e:
            raise CustomException(e, sys)