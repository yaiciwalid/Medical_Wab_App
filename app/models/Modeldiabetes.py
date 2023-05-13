from app.preprocessing.diabetes import Diabetes
import sys
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import os
from keras.models import load_model
sys.path.append('../')


class DiabeteModel:
    def __init__(self, load=False):
        """Initialize diabetes dataset preprocessing repository"""

        self._preprocessClass = Diabetes()
        self._preprocessClass.read()

        if load is True:
            self.load_model()

    @property
    def data(self):
        return self._preprocessClass

    def pre_process(self, rm_duplicates=True, normalize=False, load=False):

        """Launsh preporcess of the dataset, spliting target and data and iniating  the model"""
        self._preprocessClass.preprocess(rm_duplicates=rm_duplicates, normalize=normalize)
        self._values_data, self._values_target = self._preprocessClass.split_data_target()
        self._nb_features = self._values_data.shape[1]
        if not load:
            self.define_model()

    def define_model(self):
        """Definition of the model"""
        np.random.seed(1)
        model = keras.Sequential(layers=[
                                keras.layers.Input(shape=(None, self._nb_features), dtype=tf.float32),
                                keras.layers.Dense(128, activation='relu'),
                                keras.layers.Dense(64, activation='relu'),
                                keras.layers.Dense(32, activation='relu'),
                                keras.layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self._model = model

    def split_train_data(self, test_size=0.4):
        """Split dataset intro train and test"""
        x_train, x_test, self.y_train, self.y_test = train_test_split(self._values_data, self._values_target,
                                                                      test_size=test_size, random_state=42)
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)

    def train_model(self, epochs=100):
        """Trainging the model"""
        self._model.fit(x=self.x_train, y=self.y_train, epochs=epochs)

    def test_model(self,  threshold=0.5):
        """Testing the model with the test set"""
        y_pred = self.predict_set(self.x_test, threshold=threshold)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def predict_set(self, set, threshold=None):
        """Predect diabetes for a list of data values"""
        y_pred = self._model.predict(set)
        if threshold is None:
            return y_pred
        else:
            return (y_pred > threshold) * 1

    def save_model(self):
        """Save the trained model"""
        path = os.path.join(self._preprocessClass._model_save_path,
                            self._preprocessClass._model_last_save_path)
        self._model.save(path)

    def load_model(self):
        """Load the trained model"""
        path = os.path.join(self._preprocessClass._model_save_path,
                            self._preprocessClass._model_last_save_path)
        self._model = load_model(path)

    def predict(self, threshold=0.5, person={
        'cholesterol_level': 0,
        'glucose_level': 0,
        'hdl_cholesterol': 0,
        'cholesterol_hdl_ratio': 0,
        'age': 0,
        'gender': 0,
        'height': 0,
        'weight': 0,
        'body_mass_idx': 0,
        'systolic_blood_pressure': 0,
        'diastolic_blood_pressure': 0,
        'waist_size': 0, 'hip_size': 0,
        'waist_hip_size_ratio': 0,
                                }):
        """Predect diabetes for a person from attributes"""
        encoded_person = self.data.encode_element(person.copy())
        values_person = np.array([encoded_person[col] for col in self.data.sorted_data_columns()])
        values_person = values_person.astype(np.float32).reshape((1, self._nb_features))
        pred = self.predict_set(values_person, threshold=threshold)[0][0]
        return pred
