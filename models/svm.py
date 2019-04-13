import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.model import Model


class SVM(Model):
    def __init__(self, C=1.0, gamma='auto', useRaceRatio=True):
        self.model = SVC(C=C, gamma=gamma, probability=True)
        self.scaler = StandardScaler()
        self.useRaceRatio = useRaceRatio
    def update(self, profiles, matches):
        """
        Update model with one match
        :param profiles: profiles of players in the order of matches, in the form of [profile1, profile2]
        :param matches: list of lists in the form of [1, 0] indicating which player won
        :return: null
        """
        Xs = []
        Ys = []
        for i in range(0, len(matches)):
            features1 = profiles[i][0].getFeatures(profiles[i][1], useRaceRatio=self.useRaceRatio)
            features2 = profiles[i][1].getFeatures(profiles[i][0], useRaceRatio=self.useRaceRatio)

            # Update with profiles in both slots to prevent strange asymmetrical model
            Xs.append(features1 + features2)
            Ys.append(str(matches[i][1]))

            Xs.append(features2 + features1)
            Ys.append(str(matches[i][0]))

        self.model.fit(self.scaler.fit_transform(Xs), Ys)

    def predict(self, profile1, profile2):
        features1 = profile1.getFeatures(profile2, useRaceRatio=self.useRaceRatio)
        features2 = profile2.getFeatures(profile1, useRaceRatio=self.useRaceRatio)
        features = np.array(features1 + features2).reshape(1, -1)
        return self.model.predict_proba(self.scaler.transform(features))[0]

    def predictBatch(self, profiles):
        Xs = []
        for i in range(0, len(profiles)):
            features1 = profiles[i][0].getFeatures(profiles[i][1], useRaceRatio=self.useRaceRatio)
            features2 = profiles[i][1].getFeatures(profiles[i][0], useRaceRatio=self.useRaceRatio)

            # Update with profiles in both slots to prevent strange asymmetrical model
            Xs.append(features1 + features2)
        return self.model.predict_proba(self.scaler.transform(Xs))

    def test(self, profiles1, profiles2, matches):
        Xs = []
        Ys = []
        for i in range(0, len(matches)):
            features1 = profiles1[i].getFeatures(profiles2[i], useRaceRatio=self.useRaceRatio)
            features2 = profiles2[i].getFeatures(profiles1[i], useRaceRatio=self.useRaceRatio)

            Xs.append(features1 + features2)
            Ys.append(matches[i][0])

            # Xs.append(features2 + features1)
            # Ys.append(matches[i][1])
        print("model score", self.model.score(self.scaler.transform(Xs), Ys))

    def updateRaw(self, features, matches, valFeatures, valMatches):
        transformedMatches = [match[1] for match in matches]
        self.model.fit(self.scaler.fit_transform(features), transformedMatches)

    def predictRaw(self, features):
        features = np.array(features).reshape(1, -1)
        return self.model.predict_proba(self.scaler.transform(features))[0]

    def predictBatchRaw(self, features):
        return self.model.predict_proba(self.scaler.transform(features))

    def getFeatures(self, profile1, profile2):
        features1 = profile1.getFeatures(profile2, useRaceRatio=self.useRaceRatio)
        features2 = profile2.getFeatures(profile1, useRaceRatio=self.useRaceRatio)
        return features1 + features2