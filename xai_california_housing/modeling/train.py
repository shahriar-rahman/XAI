import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
sys.path.append(os.path.abspath('../'))
from plots import Plots
from sklearn.model_selection import GridSearchCV
import pickle


class Train:
    def __init__(self, input_path='../../dataset/processed/california_housing.csv', model_path='../../models/'):
        self.input_path = input_path
        self.model_path = model_path
        self.df = pd.read_csv(self.input_path)
        self.plot = Plots()

        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()

    def data_partition(self):
        # Split the DataFrame into train and test data
        x = self.df.drop(columns=['MedHouseVal'])
        y = self.df['MedHouseVal']

        try:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,
                                                                                    random_state=42)

        except Exception as exc:
            print(f"An error occured {exc}")

        else:
            print("Partition Successful.")

    def grid_search_cv(self):
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                                   param_grid=param_grid, cv=3, verbose=2, scoring='neg_mean_squared_error')

        grid_search.fit(self.x_train, self.y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f'Best parameters: {best_params}')
        print(f'Best score: {best_score}')

    def xgb_model(self):
        # Best Parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}
        # Initialize XGBoost regressor with the best parameters
        final_model = xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            objective='reg:squarederror',
            random_state=42
        )

        final_model.fit(self.x_train, self.y_train)

        # Save the model
        model_filename = self.model_path+'xgboost_final_model.pkl'

        with open(model_filename, 'wb') as file:
            pickle.dump(final_model, file)

        print(f"Model saved to {self.model_path+model_filename}")

        # Save the train data
        train_data_filename = self.model_path + 'train_data.pkl'

        with open(train_data_filename, 'wb') as file:
            pickle.dump((self.x_train, self.y_train), file)

        print(f"Train data saved to {train_data_filename}")

        # Save the test data
        test_data_filename = self.model_path+'test_data.pkl'

        with open(test_data_filename, 'wb') as file:
            pickle.dump((self.x_test, self.y_test), file)

        print(f"Test data saved to {test_data_filename}")


if __name__ == "__main__":
    main = Train()
    main.data_partition()
    main.grid_search_cv()
    main.xgb_model()
