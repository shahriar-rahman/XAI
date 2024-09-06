from lime import lime_tabular
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from matplotlib.pyplot import savefig


class Interpretation:
    def __init__(self, model_path='../../models/', explanation_path='../../reports/explanation/',
                 fig_path='../../reports/figures/'):
        self.explanation_path = explanation_path
        self.fig_path = fig_path
        train_data_filename = model_path+'train_data.pkl'
        test_data_filename = model_path+'test_data.pkl'
        model_filename = model_path+'xgboost_final_model.pkl'

        # Load the train data
        with open(train_data_filename, 'rb') as file:
            self.x_train_loaded, self.y_train_loaded = pickle.load(file)

        # Load the test data
        with open(test_data_filename, 'rb') as file:
            self.x_test_loaded, self.y_test_loaded = pickle.load(file)

        # Load the model
        with open(model_filename, 'rb') as file:
            self.loaded_model = pickle.load(file)

    def lime_explainer(self):
        # LIME Explainer setup
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.x_train_loaded),
            feature_names=self.x_train_loaded.columns,
            class_names=['MedHouseVal'],
            mode='regression'
        )

        # Explain a specific instance
        i = 10
        instance = self.x_test_loaded.iloc[i]

        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=self.loaded_model.predict
        )

        # Visualize explanation
        explanation.save_to_file(f'{self.explanation_path}lime_explanation_sample{str(i)}.html')
        explanation.as_pyplot_figure('MedHouseVal')
        plt.savefig(self.explanation_path+f'lime_explanation_sample{str(i)}.png')

    def shap_explainer(self):
        # SHAP Explainer setup
        explainer = shap.TreeExplainer(self.loaded_model)
        shap_values = explainer.shap_values(self.x_test_loaded)

        # Summary plot of SHAP values
        shap.summary_plot(shap_values, self.x_test_loaded, feature_names=self.x_test_loaded.columns, show=False)
        plt.savefig(f"{self.fig_path}shap_plot.png")
        plt.show()

        # Bar plot representations for SHAP values
        shap.summary_plot(shap_values, self.x_test_loaded, feature_names=self.x_test_loaded.columns,
                          plot_type='bar', show=False)
        plt.savefig(f"{self.fig_path}shap_plot_bar.png")
        plt.show()

        # Local interpretation
        i = 10

        shap_html = shap.force_plot(explainer.expected_value, shap_values[i], self.x_test_loaded.iloc[i],
                                    feature_names=self.x_test_loaded.columns)
        shap.save_html(f'{self.explanation_path}shap_explanation_sample{i}.html', shap_html, full_html=True)

    def permutation_importance(self):
        perm_importance = permutation_importance(self.loaded_model, self.x_test_loaded, self.y_test_loaded,
                                                 n_repeats=30, random_state=42)

        # Convert the permutation importance results to a sorted order
        sorted_idx = perm_importance.importances_mean.argsort()

        # Plot Permutation Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(self.x_test_loaded.columns[sorted_idx], perm_importance.importances_mean[sorted_idx], color='purple')
        plt.xlabel("Permutation Importance")
        plt.title("Permutation Feature Importance")
        plt.savefig(f"{self.fig_path}perm_plot.png")
        plt.show()


if __name__ == "__main__":
    main = Interpretation()
    main.lime_explainer()
    main.shap_explainer()
    main.permutation_importance()
