from models.model import Model
import numpy as np

class Elo(Model):
    """
    DEPRECATED NO LONGER WORKS
    """
    def update(self, profiles, matches):
        """
        Update model with one match
        :param profiles: profiles of players in the order of matches, in the form of [profile1, profile2]
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

    def predictBatch(self, profiles):
        out = []
        for i in range(len(profiles)):
            out.append(self.predict(profiles[i][0], profiles[i][1]))
        return np.array(out)

    def test(self, profiles1, profiles2, matches):
        raise NotImplementedError()