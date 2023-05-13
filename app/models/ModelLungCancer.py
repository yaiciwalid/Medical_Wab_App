from app.preprocessing.LungCancer import LungCancer
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import joblib


sys.path.append('../')


class LungCancerModel:

    def __init__(self, load=False):
        """Initialize diabetes dataset preprocessing repository"""

        self._preprocessClass = LungCancer()

        if load:
            self.load_model()
        else:
            self._preprocessClass.read()

    @property
    def data(self):
        return self._preprocessClass

    def pre_process(self, rm_duplicates=True, normalize=False, load=False):
        self._preprocessClass.preprocess(rm_duplicates=rm_duplicates, normalize=normalize)
        self._values_data, self._values_target = self._preprocessClass.split_data_target()
        self._nb_features = self._values_data.shape[1]
        if not load:
            self.define_model()

    def define_model(self):
        self.model = LinearDiscriminantAnalysis(solver='svd')

    def split_train_data(self, test_size=0.3):
        x_train, x_test, self.y_train, self.y_test = train_test_split(self._values_data, self._values_target,
                                                                      test_size=test_size, random_state=0)
        self.x_train = x_train
        self.x_test = x_test

    def train_model(self):

        self.model.fit(self.x_train, self.y_train)

    def test_model(self):
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def save_model(self):
        """Save the trained model"""
        path = os.path.join(self._preprocessClass._model_save_path,
                            self._preprocessClass._model_last_save_path)
        joblib.dump(self.model, path)

    def load_model(self):
        """Load the trained model"""
        path = os.path.join(self._preprocessClass._model_save_path,
                            self._preprocessClass._model_last_save_path)
        self.model = joblib.load(path)

    def predict(self, person={
        'Age': 45,
        'Gender': 1,
        'Air_Pollution': 0,
        'Alcohol_use': 0,
        'Dust_Allergy': 0,
        'OccuPational_Hazards': 0,
        'Genetic_Risk': 0,
        'chronic_Lung_Disease': 0,
        'Balanced_Diet': 0,
        'Obesity': 0,
        'Smoking': 0,
        'Passive_Smoker': 0,
        'Chest_Pain': 0,
        'Coughing_of_Blood': 0,
        'Fatigue': 0,
        'Weight_Loss': 0,
        'Shortness_of_Breath': 0,
        'Wheezing': 0,
        'Swallowing_Difficulty': 0,
        'Clubbing_of_Finger_Nails': 0,
        'Frequent_Cold': 0,
        'Dry_Cough': 0,
        'Snoring': 0,
                    }):
        """Predect diabetes for a person from attributes"""
        person_copy = person.copy()
        ordered_input_colmns = [x for x in self._preprocessClass._output_colmns.values() if x in person_copy.keys()]
        print(ordered_input_colmns)
        print(self._preprocessClass._output_colmns)
        values_person = np.array([person_copy[col]for col in ordered_input_colmns])
        values_person = values_person.astype(np.float32).reshape((1, len(ordered_input_colmns)))
        pred = self.model.predict(values_person)[0]
        return pred
