"""Class to preprocess diabetes data file"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from app.preprocessing.data import DataConfig


class Diabetes(DataConfig):

    def __init__(self):
        """Initialize diabetes dataset preprocessing repository"""
        super().__init__()
        self._diabetes_config = self.infra_config['diabetes']

        self._data_path = (
            os.path.join(
                os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                self._diabetes_config['data']['directory']
            )
        )

        self._file_path = (
            os.path.join(
                self._data_path,
                self._diabetes_config['data']['sub_directory'],
                self._diabetes_config['data']['file_name']
            )
        )

        self._model_save_path = (
            os.path.join(
                os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                self._diabetes_config['model']['directory'],
                self._diabetes_config['model']['sub_directory'],
            )
        )
        self._model_last_save_path = self._diabetes_config['model']['last_model_name']
        self._input_colmns = self._diabetes_config['columns']

        self._output_colmns = {}
        self._string_colmns = []
        self._colmns_dtype = {}
        self._classfication_variable = self._diabetes_config['data']['classification_column']
        self._id_column = self._diabetes_config['data']['id_column']
        self._string_colmns_encoder = {}

        # If target variable need a special transformation
        if 'classification_mapping' in self._diabetes_config['data']:
            self._target_mapping = self._diabetes_config['data']['classification_mapping']
        else:
            self._target_mapping = None
        # Saving Zmin and Zmax used to scale (normalize) data for the nextcoming data
        self._scaling_parametres = {}
        self._to_normalize_columns = []

        for col in self._input_colmns:

            # get output colmuns name
            if self._input_colmns[col]['output_column']:
                self._output_colmns[col] = self._input_colmns[col]['output_column']

            # get string columns (categorical)
            if self._input_colmns[col]['type'] == 'string':
                self._string_colmns.append(self._input_colmns[col]['output_column'])
                self._string_colmns_encoder[self._input_colmns[col]['output_column']] = LabelEncoder()
            # get columns types
            self._colmns_dtype[col] = self._input_colmns[col]['type']

            # get columns to normalize

            if self._input_colmns[col]['normalize'] == 'min-max':
                self._to_normalize_columns.append(self._input_colmns[col]['output_column'])

    @property
    def df_input(self):
        """Returns the diabetes input dataset without preprocessing"""
        return self._diabetes_df_input

    @property
    def df(self):
        """Returns the diabetes dataset after preprocessing"""
        return self._diabetes_df

    def read(self):
        """Read diabetes input dataset"""
        self._diabetes_df_input = (
            pd.read_csv(
                self._file_path,
                dtype=self._colmns_dtype,
                converters={
                    'chol_hdl_ratio': self.str_to_float,
                    'bmi': self.str_to_float,
                    'waist_hip_ratio': self.str_to_float
                },
            )
        )

    @property
    def target(self):
        return self._classfication_variable

    def remove_duplicates(self):
        """Remove duplicates from the file"""
        self._diabetes_df.drop_duplicates()

    def remove_id_column(self):
        """Remove ID column"""
        self._diabetes_df.drop(self._id_column, axis=1, inplace=True)

    def preprocess(self, rm_duplicates=False, normalize=False):
        """Preprocess diabetes dataset"""
        self._diabetes_df = self._diabetes_df_input.copy()
        self.rename_colmns()
        self.categorical_transform()
        self.remove_id_column()
        if rm_duplicates:
            self.remove_duplicates()
        if normalize:
            self.normalize_data()

    def normalize_data(self):
        """Normalize data with min-max scaling"""
        for col in self._to_normalize_columns:
            min_col = self._diabetes_df[col].min()
            max_col = self._diabetes_df[col].max()
            # If the column have diffrents values
            if max_col != min_col:
                self._diabetes_df[col] = (self._diabetes_df[col] - min_col)/(max_col - min_col)
                self._scaling_parametres[col] = [min_col, max_col]

    def categorical_transform(self):
        """Transform categorical (string) columns into integers codes"""
        for col in self._string_colmns:
            if col != self._classfication_variable:
                encoder = self._string_colmns_encoder[col]
                self._diabetes_df[col] = encoder.fit_transform(self._diabetes_df[col])
            else:
                if self._target_mapping is not None:
                    self._diabetes_df[col] = self._diabetes_df[col].map(self._target_mapping)

    def rename_colmns(self):
        """Rename input columns in diabetes dataset"""
        self._diabetes_df.rename(columns=self._output_colmns, inplace=True)
        self._classfication_variable = self._input_colmns[self._classfication_variable]['output_column']
        self._id_column = self._input_colmns[self._id_column]['output_column']

    def str_to_float(self, x):
        """Converting string to float with erorr handling"""
        return pd.to_numeric(x.replace(',', '.'), errors='coerce', downcast='float')

    def split_data_target(self):
        """Split dataframe into data and target column"""
        self._values_data = self._diabetes_df.drop([self.target], axis=1).values
        self._values_target = self._diabetes_df[self.target].values
        return self._values_data, self._values_target

    def sorted_data_columns(self):
        """Return the columns of the training data with the same order used in training"""
        return self._diabetes_df.drop([self.target], axis=1).columns

    def encode_element(self, person):
        """Apply the encoding used for a new data (a new person)"""
        for col in self._string_colmns:
            if col != self._classfication_variable:
                encoder = self._string_colmns_encoder[col]
                person[col] = encoder.transform([person[col]])[0]
        return person
