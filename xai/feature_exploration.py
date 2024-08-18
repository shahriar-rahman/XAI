import python_utils.common_utils as cu
from plots import Visualize
input_path = "../dataset/raw/california_housing.csv"
output_path = "../dataset/interim/california_housing.csv"
save_path = "../reports/figures/"


def show_on_console(text, code):
    print('\n')
    print('-' * 150)
    print("◘ ", text)
    print(code)
    print('-' * 150)


class FeatureAnalysis:
    def __init__(self):
        self.input_path = input_path
        self.save_fig = save_path
        self.show = Visualize()
        self.output_path = output_path

    def data_exploration(self):
        # Load dataset
        df = cu.load_df(ext_type="csv", library="pd", path=self.input_path, d_type=False)
        show_on_console("Dataset inspection", df.head(10).to_string())

        # Set all columns to lowercase
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        show_on_console("Variables", df.columns)

        # Describe dataset
        show_on_console("Feature description", df.describe())

        # Multivariate Outliers
        text = "Outlier plot"
        self.show.outlier(df, text, self.save_fig)

        # Multivariate distribution
        text = "Distribution plot"
        self.show.multi_kde(df, df.columns, text, path=self.save_fig)

        for feature in df.columns:
            # Univariate plots
            self.single_plots(df, feature, feature)

            # Bivariate plots
            self.compare_plots(df, feature, df.columns, feature)

        # Save dataset
        cu.save_df(df=df, ext_type='csv', path=self.output_path)

    def single_plots(self, df, x, text):
        # Histogram
        new_text = "Histogram - "+text
        self.show.hist(df, x=x, text=new_text, path=self.save_fig)

        # KDE plot
        new_text = "KDE - "+text
        self.show.kde(df,  x=x, text=new_text, path=self.save_fig)

        # Density plot
        new_text = "Density - "+text
        self.show.density(df,  x=x, text=new_text, path=self.save_fig)

    def compare_plots(self, df, x, columns, text):
        for y in columns:
            if x != y:
                # Scatter plot
                new_text = f"Scatter - {y} vs {text}"
                self.show.scatter(df, x=x, y=y, text=new_text, path=self.save_fig)

                # Residual plot
                new_text = f"Residual - {y} vs {text}"
                self.show.residual(x=df[x], y=df[y], text=new_text, x_label=x, y_label=y, path=self.save_fig)


if __name__ == "__main__":
    main = FeatureAnalysis()
    main.data_exploration()