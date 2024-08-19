import python_utils.common_utils as cu
from plots import Visualize
input_path = "../dataset/interim/california_housing.csv"
save_path = "../reports/figures/"


def show_on_console(text, code):
    print('\n')
    print('-' * 150)
    print("◘ ", text)
    print(code)
    print('-' * 150)


class FeatureCleaning:
    def __init__(self):
        self.input_path = input_path
        self.show = Visualize()
        self.save_path = save_path

    @staticmethod
    def find_outliers(df):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        outliers = df[((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr)))]
        show_on_console("number of outliers: ", str(len(outliers)))
        show_on_console("max outlier value: ", str(outliers.max()))
        show_on_console("min outlier value: ", str(outliers.min()))
        return outliers

    def feature_cleaning(self):
        # Load dataset
        df = cu.load_df(ext_type="csv", library="pd", path=self.input_path, d_type=False)
        show_on_console("Dataset inspection", df.head(10).to_string())

        # Detect outliers for all features using statistical methods
        for column in df.columns:
            show_on_console(f"{column} outliers:", self.find_outliers(df[column]))

        # Detect missing values
        show_on_console("Sum of missing values", df.isnull().sum())
        df.info()

        # Visualizing missing values
        self.show.missingno(df, kind='matrix', path=self.save_path)
        self.show.missingno(df, kind='bar', path=self.save_path)

        # Save dataset
        cu.save_df(df=df, ext_type='csv', path=self.input_path)


if __name__ == "__main__":
    main = FeatureCleaning()
    main.feature_cleaning()
