import codecs, csv
from player_profile import PlayerProfile
from random import shuffle
from scipy.stats import binom
import numpy as np

class ModelRunner:
    def __init__(self, model):
        self.model = model
        self.profiles = {}

    def runFile(self, fileName):
        with codecs.open(fileName, "r", "utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.runGame(row)

    def runGame(self, game):
        # Output prediction and run one update of model
        # Game is a dict from a row, from csvreader
        if not game['Player1'] in self.profiles:
            self.profiles[game['Player1']] = PlayerProfile(game['Player1'], game['Race1'], game['Date'])
        if not game['Player2'] in self.profiles:
            self.profiles[game['Player2']] = PlayerProfile(game['Player2'], game['Race2'], game['Date'])

        out = self.model.predict(self.profiles[game['Player1']], self.profiles[game['Player2']])
        shuffledMatches = self.shuffleGames(int(game['Score1']), int(game['Score2']))
        for match in shuffledMatches:
            # Model is updated before profiles in order to ensure prediction order is maintained
            # Since profile update happens after the game is over, whereas prediction happens before the game
            self.model.update(self.profiles[game['Player1']], self.profiles[game['Player2']], match)
            self.profiles[game['Player1']].updateProfile(game['Date'], self.profiles[game['Player2']], match)
            self.profiles[game['Player2']].updateProfile(game['Date'], self.profiles[game['Player1']], match)


    def shuffleGames(self, score1, score2):
        out = []
        for i in score1:
            out.append([1,0])
        for i in score2:
            out.append([0,1])
        shuffle(out)
        return out


    def runLive(self):
        # Grab live data from gosugamers and updates model.
        while True:
            try:
                game = self.getLive()
                self.runGame(game)
                # TODO: Write new games to file as well
            except:
                continue

    def predict(self, player1, player2):
        if player1 in self.profiles and player2 in self.profiles:
            return self.model.predict(self.profiles[player1], self.profiles[player2])
        else:
            if player1 in self.profiles:
                raise PlayerNotFoundException(player2 + " not found")
            elif player2 in self.profiles:
                raise PlayerNotFoundException(player1 + " not found")
            else:
                raise PlayerNotFoundException("Both " + player1 + " and " + player2 + " not found")

    def predictSeries(self, player1, player2, bestOf):
        odds = self.predict(player1, player2)
        assert(bestOf % 2 == 1)
        out = {}
        totalOdds1 = 0
        totalOdds2 = 0
        for i in range(0, bestOf//2 + 1):
            key1 = "{}:{}".format(str(bestOf//2 + 1), str(i))
            out[key1] = self.calcSeriesOdds(odds, bestOf//2 + 1, i)
            totalOdds1 += out[key1]

            key2 = "{}:{}".format(str(i), str(bestOf//2 + 1))
            out[key2] = self.calcSeriesOdds(odds, i, bestOf//2 + 1)
            totalOdds2 += out[key2]
        np.testing.assert_almost_equal(totalOdds1 + totalOdds2, 1.0, decimal=4)
        return out, totalOdds1, totalOdds2

    def calcAllSeries(self, singleOdds, bestOf):
        out = {}
        totalOdds1 = 0
        totalOdds2 = 0

        for i in range(0, bestOf//2 + 1):
            key1 = "{}:{}".format(str(bestOf//2 + 1), str(i))
            out[key1] = self.calcSeriesOdds(singleOdds, bestOf//2 + 1, i)
            totalOdds1 += out[key1]

            key2 = "{}:{}".format(str(i), str(bestOf//2 + 1))
            out[key2] = self.calcSeriesOdds(singleOdds, i, bestOf//2 + 1)
            totalOdds2 += out[key2]
        np.testing.assert_almost_equal(totalOdds1 + totalOdds2, 1.0, decimal=4)
        return out, totalOdds1, totalOdds2

    def calcSeriesOdds(self, singleOdds, score1, score2):
        # Calculate odds of specific score happening in a series, given odds of player1 winning one game
        totalGames = score1 + score2

        if score1 > score2:
            return binom.pmf(k=score1-1, n=totalGames-1, p=singleOdds) * singleOdds
        else:
            return binom.pmf(k=score2 - 1, n=totalGames - 1, p=1-singleOdds) * (1-singleOdds)


    def getLive(self):
        # Grab live data from gosugamers
        pass

class PlayerNotFoundException(Exception):
    pass

if __name__ == "__main__":
    runner = ModelRunner(None)
    print(runner.calcAllSeries(0.3851, 7))