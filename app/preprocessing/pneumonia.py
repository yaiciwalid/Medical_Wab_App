"""Class to preprocess diabetes data file"""

from app.preprocessing.data import DataConfig
import os
import pandas as pd
from glob import glob


class Pneumonia(DataConfig):
    def __init__(self):
        super().__init__()
        """Initialize diabetes dataset preprocessing repository"""
        self._pneumonia_config = self.infra_config['pneumonia']
        self._data_path = (
            os.path.join(
                os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                self._pneumonia_config['data']['directory']
            )
        )
        self._models_path = (
            os.path.join(
                os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                self._pneumonia_config['models']['directory'],
                self._pneumonia_config['models']['sub_directory'],
                self._pneumonia_config['models']['directory_name'],
            )
        )
        self._train_file_path = (
            os.path.join(
                self._data_path,
                self._pneumonia_config['data']['sub_directory'],
                self._pneumonia_config['data']['directory_name'],
                self._pneumonia_config['data']['train_directory'],
            )
        )
        self._test_file_path = (
            os.path.join(
                self._data_path,
                self._pneumonia_config['data']['sub_directory'],
                self._pneumonia_config['data']['directory_name'],
                self._pneumonia_config['data']['test_directory'],
            )
        )
        self._validation_file_path = (
            os.path.join(
                self._data_path,
                self._pneumonia_config['data']['sub_directory'],
                self._pneumonia_config['data']['directory_name'],
                self._pneumonia_config['data']['validation_directory'],
            )
        )
        self._classification_pneumonia_folders = self._pneumonia_config['data']['classification_pneumonia_folders']
        self._classification_normal_folders = self._pneumonia_config['data']['classification_normal_folders']

        self._pretrained_encoder_path = (
            os.path.join(
                self._models_path,
                self._pneumonia_config['models']['encoder'],
            )
        )
        self._pretrained_classifier_path = (
            os.path.join(
                self._models_path,
                self._pneumonia_config['models']['classifier'],
            )
        )

    def read(self):
        """Read pneumonia input dataset"""
        train_data = pd.DataFrame(self.read_subset(self._train_file_path), columns=['image', 'label'], index=None)
        test_data = pd.DataFrame(self.read_subset(self._test_file_path), columns=['image', 'label'], index=None)
        validation_data = pd.DataFrame(self.read_subset(self._validation_file_path),
                                       columns=['image', 'label'], index=None)
        return train_data, test_data, validation_data

    def read_subset(self, path):
        """Read a subset of the dataset from path"""
        normal_xrays = glob(os.path.join(path, self._classification_pneumonia_folders)+'/*.jpeg')
        pneumonia_xrays = glob(os.path.join(path, self._classification_normal_folders)+'/*.jpeg')
        data = [(img, 0) for img in normal_xrays] + [(img, 1) for img in pneumonia_xrays]
        return data

    def get_list_folders(self, subset='train'):
        """Get path of a subset of dataset"""
        if subset == 'train':
            return os.path.join(
                    self._train_file_path,
                            )
        # We use the training set as validation because validation is not enought
        elif subset == 'val':
            return os.path.join(self._test_file_path,)
