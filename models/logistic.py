from sklearn.linear_model import LogisticRegression
from models.model import Model
import numpy as np

class Logistic(Model):
    def __init__(self):
        self.model = LogisticRegression(solver='newton-cg', multi_class='multinomial', max_iter=1000, C=1e5)
    def update(self, profiles1, profiles2, matches):
        """
        Update model with one match
        :param profiles1: profiles of first players
        :param profiles2: profiles of second players
        :param matches: list of lists in the form of [1, 0] indicating which player won
        :return: null
        """
        Xs = []
        Ys = []
        for i in range(0, len(matches)):
            features1 = profiles1[i].getFeatures(profiles2[i])
            features2 = profiles2[i].getFeatures(profiles1[i])


            # Update with profiles in both slots to prevent strange asymmetrical model
            Xs.append(features1 + features2)
            Ys.append(matches[i][1])

            Xs.append(features2 + features1)
            Ys.append(matches[i][0])

        self.model.fit(Xs, Ys)

    def predict(self, profile1, profile2):
        features1 = profile1.getFeatures(profile2)
        features2 = profile2.getFeatures(profile1)
        features = np.array(features1 + features2).reshape(1, -1)
        return self.model.predict_proba(features)

    def predictBatch(self, profiles1, profiles2):
        Xs = []
        for i in range(0, len(profiles1)):
            features1 = profiles1[i].getFeatures(profiles2[i])
            features2 = profiles2[i].getFeatures(profiles1[i])

            # Update with profiles in both slots to prevent strange asymmetrical model
            Xs.append(features1 + features2)

            Xs.append(features2 + features1)
        return self.model.predict_proba(Xs)


    def test(self, profiles1, profiles2, matches):
        Xs = []
        Ys = []
        for i in range(0, len(matches)):
            features1 = profiles1[i].getFeatures(profiles2[i])
            features2 = profiles2[i].getFeatures(profiles1[i])

            Xs.append(features1 + features2)
            Ys.append(matches[i][0])

            # Xs.append(features2 + features1)
            # Ys.append(matches[i][1])
        print("model score", self.model.score(Xs, Ys))
