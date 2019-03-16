class PlayerProfile:
    def __init__(self, name, race, firstDate):
        self.lastPlayedDate = firstDate
        self.wins = 0
        self.winsZ = 0
        self.winsT = 0
        self.winsP = 0
        self.total = 0
        self.totalZ = 0
        self.totalT = 0
        self.totalP = 0
        self.rating = 0

    def updateProfile(self, date, opponentProfile, win):
        pass

    def getFeatures(self):
        return None