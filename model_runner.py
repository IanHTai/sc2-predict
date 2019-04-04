import codecs, csv
from player_profile import PlayerProfile
from random import shuffle
from scipy.stats import binom
import numpy as np
from copy import deepcopy
from models.linear import Linear
from models.logistic import Logistic
from models.glicko import Glicko
from models.elo import Elo
from datetime import datetime
import helper
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ModelRunner:
    def __init__(self, model, fileName, trainRatio=0.8, testRatio=0.2):
        self.model = model
        self.profiles = {}
        self.fileName = fileName
        self.inDict = {'profile1': [], 'profile2': [], 'matches': []}
        self.trainRatio = trainRatio
        self.testRatio = testRatio

    def runFile(self, fileName=None, test=False, validation=False):
        if fileName is None:
            fileName = self.fileName
        # trainLines = -1
        # testLines = -1
        if test:
            with codecs.open(fileName, "r", "utf-8") as file:
                lines = sum(1 for line in file)
            # print(lines)
            # trainLines = int(lines * self.trainRatio)
            # testLines = int(lines * self.testRatio)
            # print(trainLines)
            # print(testLines)
        with codecs.open(fileName, "r", "utf-8") as file:

            reader = csv.DictReader(file)
            for row in reader:
                matchesOut = self.runGame(row)
                self.inDict['profile1'] = self.inDict['profile1'] + matchesOut['profile1']
                self.inDict['profile2'] = self.inDict['profile2'] + matchesOut['profile2']
                self.inDict['matches'] = self.inDict['matches'] + matchesOut['matches']


    def runGame(self, game):
        # TODO: Add race into profile dict as Stats_P or Maru_T
        # Output inputDict for updating model
        # Game is a dict from a row, from csvreader
        if not game['Player1'] in self.profiles:
            race1 = None
            if 'Player1Race' in game.keys():
                race1 = game['Player1Race']
            self.profiles[game['Player1']] = PlayerProfile(game['Player1'], race1, game['Date'], game['Player1Region'])
        if not game['Player2'] in self.profiles:
            race2 = None
            if 'Player2Race' in game.keys():
                race2 = game['Player2Race']
            self.profiles[game['Player2']] = PlayerProfile(game['Player2'], race2, game['Date'], game['Player2Region'])

        out = {'profile1': [], 'profile2': [], 'matches': []}
        shuffledMatches = self.shuffleGames(int(game['Score1']), int(game['Score2']))


        for match in shuffledMatches:
            # Model is updated before profiles in order to ensure prediction order is maintained
            # Since profile update happens after the game is over, whereas prediction happens before the game
            # self.model.update(self.profiles[game['Player1']], self.profiles[game['Player2']], match)

            # Check decay
            date = datetime.strptime(game['Date'], self.profiles[game['Player1']].dateFormat).date()
            self.profiles[game['Player1']].checkDecay(date)
            self.profiles[game['Player2']].checkDecay(date)

            out['profile1'].append(deepcopy(self.profiles[game['Player1']]))
            out['profile2'].append(deepcopy(self.profiles[game['Player2']]))
            out['matches'].append(match)

            self.profiles[game['Player1']].updateProfile(game['Date'], self.profiles[game['Player2']], match[0] == 1)
            self.profiles[game['Player2']].updateProfile(game['Date'], self.profiles[game['Player1']], match[1] == 1)

        return out

    def createTTV(self, trainPercentage=0.7, testPercentage=0.15):
        # Create train, test, and validate
        inArr = []
        matchArr = []
        for (player1, player2, match) in zip(self.inDict['profile1'], self.inDict['profile2'], self.inDict['matches']):
            inArr.append(np.array([player1, player2]))
            matchArr.append(np.array(match))
        inArr = np.array(inArr)
        matchArr = np.array(matchArr)

        print(inArr.shape)
        print(matchArr.shape)

        self.train_X, test_x, self.train_Y, test_y = train_test_split(inArr, matchArr, test_size=1 - trainPercentage)
        print(self.train_X.shape, self.train_Y.shape, test_x.shape, test_y.shape)
        assert(len(test_x) == len(test_y))

        self.test_X, self.val_X, self.test_Y, self.val_Y = train_test_split(test_x, test_y, test_size=(1 - trainPercentage - testPercentage)/(1 - trainPercentage))
        assert(len(self.test_X) == len(self.test_Y))

    def updateModel(self):
        # Must be called after createTTV
        self.model.update(self.train_X, self.train_Y)

    def testModel(self):
        # Must be called after createTTV
        predictions = self.model.predictBatch(self.test_X)
        print(predictions)
        real = self.test_Y
        total = 0

        print(real)
        for i in range(len(real)):
            #total += predictions[i] * real[i]
            total += abs(real[i] - predictions[i])
        print(total, len(real), total/len(real))

        self.stats(predictions, real)

        return total/len(real)

    def stats(self, preds, real):
        linspace = np.linspace(0.5, 1.0, 25)
        buckets = np.zeros((25,2))
        # buckets from 50 to 100
        real = np.array(real)
        print("predshape", preds.shape, "realshape", real.shape)
        assert(len(preds) == len(real))
        for i in range(0, len(real)):
            matchIndex = np.argmax(preds[i])
            bucketIndex = int(max(preds[i][0], preds[i][1]) * 50) - 25
            if bucketIndex == 20:
                # If 100% prediction
                bucketIndex = 19
            buckets[bucketIndex][0] += real[i][matchIndex]
            buckets[bucketIndex][1] += 1

        bucketResults = np.divide(buckets[:,0], buckets[:,1])# buckets[:,0]/buckets[:,1]
        plt.plot(linspace, linspace, 'r--', label='perfect predictions')
        plt.plot(linspace, bucketResults, 'bo', label='actual predictions')
        idx = np.isfinite(linspace) & np.isfinite(bucketResults)
        plt.plot(linspace, np.poly1d(np.polyfit(linspace[idx], bucketResults[idx], 1))(linspace), 'b--', label='fitted to predictions')
        #plt.plot(np.unique(linspace), np.poly1d(np.polyfit(linspace, bucketResults, 1))(np.unique(linspace)), )
        #plt.plot(linspace, np.polynomial.polynomial.Polynomial.fit(linspace, bucketResults, 1), 'b--', label='fitted predictions')

        plt.xlabel("Predicted winrate")
        plt.ylabel("Actual winrate")
        plt.title("Actual vs Predicted Winrate")
        plt.legend()
        plt.show()

    def shuffleGames(self, score1, score2):
        out = []
        for i in range(0, score1):
            out.append([1,0])
        for i in range(0, score2):
            out.append([0,1])
        shuffle(out)
        return out


    def runLive(self):
        # Grab live data from gosugamers and updates model.
        while True:
            try:
                game = self.getLive()
                matchesOut = self.runGame(game)
                self.inDict['profile1'] = self.inDict['profile1'] + matchesOut['profile1']
                self.inDict['profile2'] = self.inDict['profile2'] + matchesOut['profile2']
                self.inDict['matches'] = self.inDict['matches'] + matchesOut['matches']
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
    helper.fillRegionDict()
    #model = Linear()
    #model = Glicko()
    #model = Elo()
    model = Logistic()
    print('Model Created')
    runner = ModelRunner(model, "data/matchResults_regionsRaces.csv", trainRatio=0.8, testRatio=0.2)
    print('Model Runner Created')
    runner.runFile(test=True)
    print('File Run')
    runner.createTTV(0.7, 0.2)
    print('TTV Separated')
    runner.updateModel()
    print('Model Updated')
    runner.testModel()

    # for key in runner.profiles.keys():
    #     runner.profiles[key].checkDecay(datetime.now().date())

    rank = 1
    for name in sorted(runner.profiles, key=lambda name: runner.profiles[name].elo, reverse=True):
        timeSinceFirst = (datetime.now().date() - runner.profiles[name].firstPlayedDate).days
        print(rank, runner.profiles[name].name, runner.profiles[name].total, runner.profiles[name].glickoRating, timeSinceFirst, runner.profiles[name].total / timeSinceFirst, runner.profiles[name].elo)
        rank += 1

    print("Serral's Match History", "EXPOVERALL:", runner.profiles['Serral'].expOverall, "WINPERCENTAGE:", runner.profiles['Serral'].wins / runner.profiles['Serral'].total)
    print("Maru's Match History", "EXPOVERALL:", runner.profiles['Maru'].expOverall, "WINPERCENTAGE:",
          runner.profiles['Maru'].wins / runner.profiles['Maru'].total)