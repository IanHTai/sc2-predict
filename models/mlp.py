#TODO: KERAS
from models.model import Model
import numpy as np
from sklearn.preprocessing import StandardScaler
class MLP(Model):
    def update(self, profiles, matches):
        """
        Update model with one match
        :param profiles: profiles of players in the order of matches, in the form of [profile1, profile2]
        :param matches: list of lists in the form of [1, 0] indicating which player won
        :return: null
        """
        raise NotImplementedError()

    def predict(self, profile1, profile2):
        raise NotImplementedError()

    def predictBatch(self, profiles):
        raise NotImplementedError()

    def test(self, profiles1, profiles2, matches):
        raise NotImplementedError()