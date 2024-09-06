import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import os
import sys
sys.path.append(os.path.abspath('../'))
from plots import Plots
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt


class Predict:
    def __init__(self, input_path='../../dataset/processed/california_housing.csv', model_path='../../models/'):
        self.input_path = input_path
        self.model_path = model_path
        self.df = pd.read_csv(self.input_path)
        self.plot = Plots()
        self.loaded_rmse = 0
        self.loaded_r2 = 0

    def predict_model(self):
        test_data_filename = self.model_path+'test_data.pkl'
        model_filename = self.model_path+'xgboost_final_model.pkl'

        # Load the test data
        with open(test_data_filename, 'rb') as file:
            x_test_loaded, y_test_loaded = pickle.load(file)

        # Load the model
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)

        y_pred_loaded = loaded_model.predict(x_test_loaded)

        # Evaluate the loaded model
        self.loaded_rmse = root_mean_squared_error(y_test_loaded, y_pred_loaded)
        self.loaded_r2 = r2_score(y_test_loaded, y_pred_loaded)

        print(f'Loaded Model RMSE: {self.loaded_rmse}')
        print(f'Loaded Model R^2: {self.loaded_r2}')

    def plot_loss_metrics(self):
        # RMSE and R² values
        metrics = {'RMSE': self.loaded_rmse, 'R²': self.loaded_r2}
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # Plotting RMSE and R²
        plt.figure(figsize=(12, 6))

        # RMSE Plot
        plt.subplot(1, 2, 1)
        plt.bar('RMSE', metrics['RMSE'], color='maroon')
        plt.title('Model RMSE')
        plt.ylabel('RMSE')

        # R² Plot
        plt.subplot(1, 2, 2)
        plt.bar('R²', metrics['R²'], color='cyan')
        plt.title('Model R²')
        plt.ylabel('R²')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main = Predict()
    main.predict_model()
    main.plot_loss_metrics()
