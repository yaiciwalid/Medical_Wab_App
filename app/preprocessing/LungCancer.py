"""Class to preprocess Lung Cancer data file"""
import os
import pandas as pd
from app.preprocessing.data import DataConfig


class LungCancer(DataConfig):

    def __init__(self):
        """Initialize Lung dataset preprocessing repository"""
        super().__init__()
        self._lung_config = self.infra_config['LungCancer']

        self._data_path = (
            os.path.join(
                os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                self._lung_config['data']['directory']
            )
        )

        self._file_path = (
            os.path.join(
                self._data_path,
                self._lung_config['data']['sub_directory'],
                self._lung_config['data']['file_name']
            )
        )
        self._model_save_path = (
            os.path.join(
                os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                self._lung_config['model']['directory'],
                self._lung_config['model']['sub_directory'],
            )
        )
        self._model_last_save_path = self._lung_config['model']['last_model_name']

        self._input_colmns = self._lung_config['columns']

        self._output_colmns = {}
        self._string_colmns = []
        self._colmns_dtype = {}
        self._classfication_variable = self._lung_config['data']['classification_column']
        self._id_column = self._lung_config['data']['id_column']
        self._index_column = self._lung_config['data']['index_column']

        # If target variable need a special transformation
        if 'classification_mapping' in self._lung_config['data']:
            self._target_mapping = self._lung_config['data']['classification_mapping']
        else:
            self._target_mapping = None

        # Saving Zmin and Zmax used to scale (normalize) data for the nextcoming data
        self._scaling_parametres = {}
        self._to_noramlize_columns = []

        for col in self._input_colmns:

            # get output colmuns name
            if self._input_colmns[col]['output_column']:
                self._output_colmns[col] = self._input_colmns[col]['output_column']

            # get string columns (categorical)
            if self._input_colmns[col]['type'] == 'string':
                self._string_colmns.append(self._input_colmns[col]['output_column'])

            # get columns types
            self._colmns_dtype[col] = self._input_colmns[col]['type']

            # get columns to normalize

            if self._input_colmns[col]['normalize'] == 'min-max':
                self._to_noramlize_columns.append(self._input_colmns[col]['output_column'])

    @property
    def df_input(self):
        """Returns the lung input dataset without preprocessing"""
        return self._lung_df_input

    @property
    def df(self):
        """Returns the lung dataset after preprocessing"""
        return self._lung_df

    def read(self):
        """Read lung input dataset"""
        self._lung_df_input = (
            pd.read_csv(
                self._file_path,
                dtype=self._colmns_dtype,
            )
        )

    @property
    def target(self):
        return self._classfication_variable

    def remove_duplicates(self):
        """Remove duplicates from the file"""
        self._lung_df.drop_duplicates()

    def remove_id_index_columns(self):
        """Remove ID column"""
        self._lung_df.drop(self._id_column, axis=1, inplace=True)
        self._lung_df.drop(self._index_column, axis=1, inplace=True)

    def preprocess(self, rm_duplicates=True, normalize=False):
        """Preprocess lung dataset"""
        self._lung_df = self._lung_df_input.copy()
        self.rename_colmns()
        self.target_transform_to_int()
        self.remove_id_index_columns()
        if rm_duplicates:
            self.remove_duplicates()
        if normalize:
            self.noramlize_data()

    def noramlize_data(self):
        """Noramlize data with min-max scaling"""
        for col in self._to_noramlize_columns:
            min_col = self._lung_df[col].min()
            max_col = self._lung_df[col].max()
            # If the column have diffrents values
            if max_col != min_col:
                self._lung_df[col] = (self._lung_df[col] - min_col)/(max_col - min_col)
                self._scaling_parametres[col] = [min_col, max_col]

    def target_transform_to_int(self):
        """Transform categorical (string) columns into integers codes"""
        self._lung_df[self._classfication_variable] = self._lung_df[
                                                      self._classfication_variable].map(self._target_mapping)

    def rename_colmns(self):
        """Rename input columns in lung dataset"""
        self._lung_df.rename(columns=self._output_colmns, inplace=True)
        self._classfication_variable = self._input_colmns[self._classfication_variable]['output_column']
        self._id_column = self._input_colmns[self._id_column]['output_column']
        self._index_column = self._input_colmns[self._index_column]['output_column']

    def split_data_target(self):
        """Split dataframe into data and target column"""
        self._values_data = self._lung_df.drop([self.target], axis=1).values
        self._values_target = self._lung_df[self.target].values
        return self._values_data, self._values_target

    def sorted_data_columns(self):
        """Return the columns of the training data with the same order used in training"""
        return self._lung_df.drop([self.target], axis=1).columns
