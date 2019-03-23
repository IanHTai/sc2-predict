class Model:
    def update(self, profiles1, profiles2, matches):
        """
        Update model with one match
        :param profiles1: profiles of first players
        :param profiles2: profiles of second players
        :param matches: list of lists in the form of [1, 0] indicating which player won
        :return: null
        """
        raise NotImplementedError()

    def predict(self, profile1, profile2):
        raise NotImplementedError()

    def predictBatch(self, profiles1, profiles2):
        raise NotImplementedError()

    def test(self, profiles1, profiles2, matches):
        raise NotImplementedError()