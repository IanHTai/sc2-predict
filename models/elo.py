from models.model import Model
import numpy as np

class Elo(Model):
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
        Q_A = 10 ** (profile1.elo / 400)
        Q_B = 10 ** (profile2.elo / 400)
        E_A = Q_A / (Q_A + Q_B)
        E_B = Q_B / (Q_A + Q_B)
        return [E_A, E_B]

    def predictBatch(self, profiles1, profiles2):
        out = []
        for i in range(len(profiles1)):
            out.append(self.predict(profiles1[i], profiles2[i]))
        return np.array(out)

    def test(self, profiles1, profiles2, matches):
        raise NotImplementedError()