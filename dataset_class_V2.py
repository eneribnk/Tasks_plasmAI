import torch
import json
from torch.utils.data import Dataset
import pandas as pd


class MergedDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, sep=';')
        # normalize the data with min-max
        '''for column in df.columns:
            df[column] = (df[column] - df[column].min()) / ( df[column].max() - df[column].min())'''
        # normalize with mean, std and store the statistics for later use
        stats = dict()
        for column in df.columns:
            mean, std = df[column].mean(), df[column].std()
            df[column] = (df[column] - mean) / (std + 1e-10)  # 1e-10 is to avoid division with 0
            stats[column] = {'mean': mean, 'std': std}

        # Write statistics to a JSON file and store it
        with open('column_stats02_rand.json', 'w') as f:
            json.dump(stats, f)

        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df.columns if col not in X_columns]  # To be predicted
        self.X, self.y = df[X_columns], df[y_columns]

        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def print_column_stats(self):
        for column in self.df.columns:
            col_data = self.df[column]  # save the column data, df.columns shows the column headers

            # statistics calculations
            nan_percentage = col_data.isna().mean() * 100  # True indicates the presence of null or missing values and False indicates otherwise, from pandas
            min_value = col_data.min()
            max_value = col_data.max()
            mean_value = col_data.mean()
            std_value = col_data.std()

            # now we print the above
            print(f"Column: {column}")
            print(f"nan_percentage: {nan_percentage:.2f}%")
            print(f"Max value: {max_value}")
            print(f"Min value: {min_value}")
            print(f"average value: {mean_value}")
            print(f"std: {std_value}")
            print("-" * 30)  # print a line -----------

class MergedDatasetTest(Dataset):
    def __init__(self, csv_file, stats_file):
        df = pd.read_csv(csv_file, sep=';')
        # Load the pre-calculated mean and std statistics from the JSON file
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        for column in df.columns:
            if column in stats:
                mean = stats[column]['mean']
                std = stats[column]['std']
                df[column] = (df[column] - mean) / (std + 1e-10)  # 1e-10 to avoid division by zero

        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df.columns if col not in X_columns]  # To be predicted
        self.X, self.y = df[X_columns], df[y_columns]

        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def print_column_stats(self):
        for column in self.df.columns:
            col_data = self.df[column]  # save the column data, df.columns shows the column headers

            # statistics calculations
            nan_percentage = col_data.isna().mean() * 100  # True indicates the presence of null or missing values and False indicates otherwise, from pandas
            min_value = col_data.min()
            max_value = col_data.max()
            mean_value = col_data.mean()
            std_value = col_data.std()

            # now we print the above
            print(f"Column: {column}")
            print(f"nan_percentage: {nan_percentage:.2f}%")
            print(f"Max value: {max_value}")
            print(f"Min value: {min_value}")
            print(f"average value: {mean_value}")
            print(f"std: {std_value}")
            print("-" * 30)  # print a line -----------

class testdataset_unormalized(Dataset):
    def __init__(self):
        df = pd.read_csv('train_data_no_head_outer_corner.csv', sep=';')
        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df.columns if col not in X_columns]  # To be predicted
        self.X, self.y = df[X_columns], df[y_columns]

        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

class MergedDatasetTest_min_max(Dataset):
    def __init__(self, csv_file, stats_file='column_stats02_min_max.json'):  #if min max normalization was used in training
        df = pd.read_csv(csv_file, sep=';')
        # Load the pre-calculated mean and std statistics from the JSON file
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        for column in df.columns:
            if column in stats:
                col_min = stats[column]['min']
                col_max = stats[column]['max']
                df[column] = (df[column] - col_min) / (col_max - col_min + 1e-10)  # Avoid division by zero

        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df.columns if col not in X_columns]  # To be predicted
        self.X, self.y = df[X_columns], df[y_columns]

        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]