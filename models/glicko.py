from models.model import Model
import numpy as np

class Glicko(Model):
    def __init__(self):
        pass

    def update(self, profiles1, profiles2, matches):
        """
        Update model with one match
        :param profiles1: profiles of first players
        :param profiles2: profiles of second players
        :param matches: list of lists in the form of [1, 0] indicating which player won
        :return: null
        """
        pass

    def predict(self, profile1, profile2):

        mu_1 = (profile1.glickoRating - 1500) / 173.7178
        mu_2 = (profile2.glickoRating - 1500) / 173.7178

        phi_j = profile2.glickoRD / 173.7178
        phi_i = profile1.glickoRD / 173.7178

        return [profile1.glickoExpected(mu_1, mu_2, phi_j), profile2.glickoExpected(mu_2, mu_1, phi_i)]

    def predictBatch(self, profiles1, profiles2):
        out = []
        for i in range(len(profiles1)):
            out.append(self.predict(profiles1[i], profiles2[i]))
        return np.array(out)
    def test(self, profiles1, profiles2, matches):
        raise NotImplementedError()