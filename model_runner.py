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

class ModelRunner:
    def __init__(self, model, fileName, trainRatio=0.8, testRatio=0.2):
        self.model = model
        self.profiles = {}
        self.fileName = fileName
        self.inDict = {'profile1': [], 'profile2': [], 'matches': []}
        self.trainRatio = trainRatio
        self.testRatio = testRatio
        self.testDict = {'profile1': [], 'profile2': [], 'matches': []}
        self.validationDict = {'profile1': [], 'profile2': [], 'matches': []}

    def runFile(self, fileName=None, test=False, validation=False):
        if fileName is None:
            fileName = self.fileName
        trainLines = -1
        testLines = -1
        if test:
            with codecs.open(fileName, "r", "utf-8") as file:
                lines = sum(1 for line in file)
            print(lines)
            trainLines = int(lines * self.trainRatio)
            testLines = int(lines * self.testRatio)
            print(trainLines)
            print(testLines)
        with codecs.open(fileName, "r", "utf-8") as file:
            lineCount = 0
            reader = csv.DictReader(file)
            for row in reader:
                matchesOut = self.runGame(row)
                lineCount += 1
                if lineCount < trainLines or trainLines < 0:
                    self.inDict['profile1'] = self.inDict['profile1'] + matchesOut['profile1']
                    self.inDict['profile2'] = self.inDict['profile2'] + matchesOut['profile2']
                    self.inDict['matches'] = self.inDict['matches'] + matchesOut['matches']
                elif lineCount < testLines + trainLines or testLines < 0:
                    self.testDict['profile1'] = self.testDict['profile1'] + matchesOut['profile1']
                    self.testDict['profile2'] = self.testDict['profile2'] + matchesOut['profile2']
                    self.testDict['matches'] = self.testDict['matches'] + matchesOut['matches']
                else:
                    self.validationDict['profile1'] = self.validationDict['profile1'] + matchesOut['profile1']
                    self.validationDict['profile2'] = self.validationDict['profile2'] + matchesOut['profile2']
                    self.validationDict['matches'] = self.validationDict['matches'] + matchesOut['matches']

    def runGame(self, game):
        # TODO: Add race into profile dict as Stats_P or Maru_T
        # Output inputDict for updating model
        # Game is a dict from a row, from csvreader
        if not game['Player1'] in self.profiles:
            race1 = None
            if 'Race1' in game.keys():
                race1 = game['Race1']
            self.profiles[game['Player1']] = PlayerProfile(game['Player1'], race1, game['Date'])
        if not game['Player2'] in self.profiles:
            race2 = None
            if 'Race2' in game.keys():
                race2 = game['Race2']
            self.profiles[game['Player2']] = PlayerProfile(game['Player2'], race2, game['Date'])

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
            self.profiles[game['Player2']].updateProfile(game['Date'], self.profiles[game['Player1']], match[0] == 1)

        return out

    def updateModel(self):
        self.model.update(self.inDict['profile1'], self.inDict['profile2'], self.inDict['matches'])

    def testModel(self):
        predictions = self.model.predictBatch(self.testDict['profile1'], self.testDict['profile2'])
        print(predictions)
        real = self.testDict['matches']
        total = 0

        print(real)
        for i in range(len(real)):
            #total += predictions[i] * real[i]
            total += abs(real[i] - predictions[i])
        print(total, len(real), total/len(real))


        return total/len(real)

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
    #model = Linear()
    model = Glicko()
    #model = Elo()
    print('Model Created')
    runner = ModelRunner(model, "data/matchResults_regionsRaces.csv", trainRatio=0.8, testRatio=0.2)
    print('Model Runner Created')
    runner.runFile(test=True)
    print('File Run')
    runner.updateModel()
    print('Model Updated')
    runner.testModel()

    print(runner.predict('Serral', 'MacSed'))
    rank = 1
    for name in sorted(runner.profiles, key=lambda name: runner.profiles[name].elo, reverse=True):
        timeSinceFirst = (datetime.now().date() - runner.profiles[name].firstPlayedDate).days
        print(rank, runner.profiles[name].name, runner.profiles[name].total, runner.profiles[name].glickoRating, timeSinceFirst, runner.profiles[name].total / timeSinceFirst, runner.profiles[name].elo)
        rank += 1

    print("Serral's Match History", "EXPOVERALL:", runner.profiles['Serral'].expOverall, "WINPERCENTAGE:", runner.profiles['Serral'].wins / runner.profiles['Serral'].total)
    print("Maru's Match History", "EXPOVERALL:", runner.profiles['Maru'].expOverall, "WINPERCENTAGE:",
          runner.profiles['Maru'].wins / runner.profiles['Maru'].total)