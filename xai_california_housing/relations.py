import pandas as pd
from plots import Plots
from matplotlib.pyplot import savefig


class Relations:
    def __init__(self, input_path='../dataset/processed/california_housing.csv', fig_path='../reports/figures/'):
        self.input_path = input_path
        self.fig_path = fig_path
        self.df = pd.read_csv(self.input_path)
        self.plot = Plots()

        self.predictors = self.df.drop(columns=['MedHouseVal']).columns.tolist()
        self.response = 'MedHouseVal'

    def linear_interpolation(self):
        # Initiate Linear Interpolation
        for predictor in self.predictors:
            self.plot.scatter(self.df, predictor, self.response, self.fig_path)

    def correlation_heatmap(self):
        # Plot the Pearson correlation coefficient
        self.plot.correlation(self.df, self.fig_path)

    def residual_plot(self):
        # Illustrate the residuals
        for predictor in self.predictors:
            self.plot.residplot(self.df, predictor, self.response, self.fig_path)


if __name__ == "__main__":
    main = Relations()
    main.linear_interpolation()
    main.correlation_heatmap()
    main.residual_plot()
