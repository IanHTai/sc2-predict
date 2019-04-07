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
from models.svm import SVM
from datetime import datetime
import helper
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats
from queue import Queue
from crawler.gosu_crawler import Crawler
import threading
from crawler.lootbet_crawler import LootCrawler, LootMatch
import time
import random

class ModelRunner:
    def __init__(self, model, fileName, lastGameId, trainRatio=0.8, testRatio=0.2, startCash = 10000):
        self.model = model
        self.profiles = {}
        self.fileName = fileName
        self.inDict = {'profile1': [], 'profile2': [], 'matches': []}
        self.trainRatio = trainRatio
        self.testRatio = testRatio
        self.profileUpdateQueue = Queue()
        self.cash = startCash
        self.startcash = startCash
        self.lastGameId = lastGameId

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

        if self.profiles[game['Player1']].race is None:
            if 'Player1Race' in game.keys():
                race1 = game['Player1Race']
                self.profiles[game['Player1']].race = race1

        if self.profiles[game['Player2']].race is None:
            if 'Player2Race' in game.keys():
                race2 = game['Player2Race']
                self.profiles[game['Player2']].race = race2

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

        if trainPercentage == 1.0:
            self.train_X = inArr
            self.train_Y = matchArr
            return

        self.train_X, test_x, self.train_Y, test_y = train_test_split(inArr, matchArr, train_size=trainPercentage)
        print(self.train_X.shape, self.train_Y.shape, test_x.shape, test_y.shape)
        assert(len(test_x) == len(test_y))


        self.test_X, self.val_X, self.test_Y, self.val_Y = train_test_split(test_x, test_y, train_size=(testPercentage)/(1 - trainPercentage))
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

        stats = self.stats(predictions, real)
        print("MSE", stats)

        return total/len(real)

    def stats(self, preds, real):
        linspace = np.linspace(0.5, 1.0, 25)
        buckets = np.zeros((25,2))
        bucketTotals = [[] for i in range(0,25)]
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
            bucketTotals[bucketIndex].append(real[i][matchIndex])

        bucketResults = np.divide(buckets[:, 0], buckets[:, 1])  # buckets[:,0]/buckets[:,1]

        def confidenceIntervals(a, results, confidence=0.95):
            outLow = []
            outHigh = []
            for i in range(len(a)):
                data = a[i]
                n = len(data)
                m, se = np.mean(data), scipy.stats.sem(data)
                h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
                outLow.append(results[i]-h)
                outHigh.append(results[i]+h)
            return outLow, outHigh

        confLow, confHigh = confidenceIntervals(bucketTotals, bucketResults)


        # plt.plot(linspace, linspace, 'r--', label='perfect predictions')
        # plt.plot(linspace, bucketResults, 'k.', label='actual predictions')
        # plt.plot(linspace, bucketResults, 'k')
        # plt.fill_between(linspace, confLow, confHigh, color='#539caf', alpha=0.4, label='95% CI')
        # idx = np.isfinite(linspace) & np.isfinite(bucketResults)
        # plt.plot(linspace, np.poly1d(np.polyfit(linspace[idx], bucketResults[idx], 1))(linspace), 'b--', label='fitted to predictions')
        # #plt.plot(np.unique(linspace), np.poly1d(np.polyfit(linspace, bucketResults, 1))(np.unique(linspace)), )
        # #plt.plot(linspace, np.polynomial.polynomial.Polynomial.fit(linspace, bucketResults, 1), 'b--', label='fitted predictions')
        #
        # plt.xlabel("Predicted winrate")
        # plt.ylabel("Actual winrate")
        # plt.title("Actual vs Predicted Winrate")
        # plt.legend()
        # plt.show()

        return np.sum(np.square(preds[:, 0] - real[:, 0]))/len(preds)

    def shuffleGames(self, score1, score2):
        out = []
        for i in range(0, score1):
            out.append([1,0])
        for i in range(0, score2):
            out.append([0,1])
        shuffle(out)
        return out

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


    def calcSeriesOdds(self, singleOdds, score1, score2):
        # Calculate odds of specific score happening in a series, given odds of player1 winning one game
        totalGames = score1 + score2
        singleOdds = singleOdds[0]

        if score1 > score2:
            return binom.pmf(k=score1-1, n=totalGames-1, p=singleOdds) * singleOdds
        else:
            return binom.pmf(k=score2 - 1, n=totalGames - 1, p=1-singleOdds) * (1-singleOdds)


    def getLive(self):
        # Grab live data from gosugamers and put it into the update queue
        url = "https://www.gosugamers.net/starcraft2/matches/results?sortBy=date-asc&maxResults=18"
        c = Crawler(url, fileName="data/matchResults_regionsRaces.csv")
        # c.start()
        liveGen = c.liveGenerator(fromPage=593, lastGameId=self.lastGameId)
        for i in liveGen:
            self.profileUpdateQueue.put(i)

    def runLive(self):
        crawlerThread = threading.Thread(target=self.getLive)
        crawlerThread.start()
        flag = False
        lc = LootCrawler(url="https://loot.bet/sport/esports/starcraft",
                         gosuUrl="https://www.gosugamers.net/starcraft2/matches")
        betOnMatches = {}
        # TODO: Proper Lootbet -> Gosu name matching for people like Dark -> Dark.Sc2, sOs -> sOs.sc2
        # TODO: Periodically update model, maybe keep track of number of new matches since last update
        while not flag:
            while not self.profileUpdateQueue.empty():
                rawMatch = self.profileUpdateQueue.get()
                if rawMatch['Id'] in betOnMatches:
                    print("Result:", rawMatch['Date'], rawMatch['Player1'], rawMatch['Player2'])
                    if int(rawMatch['Score1']) > int(rawMatch['Score2']):
                        if betOnMatches[rawMatch['Id']][0] == rawMatch['Player1']:
                            self.cash += betOnMatches[rawMatch['Id']][1]
                            print("Win:", betOnMatches[rawMatch['Id']][1])
                        else:
                            print("Lose:", betOnMatches[rawMatch['Id']][1])
                    elif int(rawMatch['Score1']) < int(rawMatch['Score2']):
                        if betOnMatches[rawMatch['Id']][0] == rawMatch['Player2']:
                            self.cash += betOnMatches[rawMatch['Id']][1]
                            print("Win:", betOnMatches[rawMatch['Id']][1])
                        else:
                            print("Lose:", betOnMatches[rawMatch['Id']][1])
                    else:
                        print("TIE OR DRAW???", rawMatch['Id'], rawMatch['Player1'], rawMatch['Player2'])
                        # Assume refund?
                        self.cash += betOnMatches[rawMatch['Id']][2]
                        print("Refund:", betOnMatches[rawMatch['Id']][2])
                    betOnMatches.pop(rawMatch['Id'])
                    print("New Balance:", self.cash, "ROI Since Start:", (self.cash - self.startcash)/float(self.startcash))

                matchesOut = self.runGame(rawMatch)
                self.inDict['profile1'] = self.inDict['profile1'] + matchesOut['profile1']
                self.inDict['profile2'] = self.inDict['profile2'] + matchesOut['profile2']
                self.inDict['matches'] = self.inDict['matches'] + matchesOut['matches']


            matches = lc.getMatches()
            if len(matches) > 0:
                for match in matches:
                    if not match.id in betOnMatches:

                        _, p1win, p2win = self.predictSeries(match.player1, match.player2, match.bestOf)

                        if p1win*match.odds1 > 1. or p2win*match.odds2 > 1.:
                            if p1win*match.odds1 > 1. and p2win*match.odds2 > 1.:
                                print("Both sides of the bet are >1? Something must have gone wrong with the bookie? Or with our model?")
                                print("P1", p1win, "Odds1", match.odds1, "P2", p2win, "Odds2", match.odds2)
                            EV = [p1win*match.odds1, p2win*match.odds2]
                            betterBet = np.argmax(EV)
                            if betterBet == 0:
                                dec = self.betDecision(prob=p1win, odds=match.odds1)
                                decOdds = dec*match.odds1
                                print("Bet:", match.player1, "vs", match.player2, "Amount:", dec, "Edge+1:",
                                      p1win * match.odds1, "Prob", p1win)
                                betOn = match.player1
                            else:
                                dec = self.betDecision(prob=p2win, odds=match.odds2)
                                decOdds = dec * match.odds2
                                print("Bet:", match.player2, "vs", match.player1, "Amount:", dec, "Edge+1:",
                                      p2win * match.odds2, "Prob", p2win)
                                betOn = match.player2
                            if dec > 0:
                                betOnMatches[match.id] = [betOn, decOdds, dec]
                                self.cash -= dec

            randomizer = random.uniform(0.8, 1.2)
            time.sleep(60 * 30 * randomizer)


    def betDecision(self, odds, prob):
        # Decide on how much to bet using Kelly Criterion

        wagerFraction = (prob*odds - 1)/(odds - 1)
        amount = wagerFraction * self.cash
        if amount <= 5:
            return 0
        else:
            return amount

class PlayerNotFoundException(Exception):
    pass


if __name__ == "__main__":
    helper.fillRegionDict()
    #model = Linear()
    #model = Glicko()
    #model = Elo()
    model = Logistic()
    #model = SVM(C=10)
    print('Model Created')
    runner = ModelRunner(model, "data/matchResults_regionsRaces.csv", trainRatio=0.8, testRatio=0.2, lastGameId="298482")
    print('Model Runner Created')
    runner.runFile(test=True)
    print('File Run')
    runner.createTTV(0.6, 0.2)

    print('TTV Separated')
    runner.updateModel()
    print('Model Updated')
    runner.testModel()

    maruAliveResults = runner.predict("Maru", "aLive")
    print("Maru", maruAliveResults[0], "aLive", maruAliveResults[1])
    print(runner.predictSeries("Maru","aLive",1))

    # for key in runner.profiles.keys():
    #     runner.profiles[key].checkDecay(datetime.now().date())

    rank = 1
    for name in sorted(runner.profiles, key=lambda name: runner.profiles[name].elo, reverse=True):
        timeSinceFirst = (datetime.now().date() - runner.profiles[name].firstPlayedDate).days
        print(rank, runner.profiles[name].name, runner.profiles[name].total, runner.profiles[name].glickoRating, timeSinceFirst, runner.profiles[name].total / timeSinceFirst, runner.profiles[name].elo,
              runner.profiles[name].eloZ, runner.profiles[name].eloT, runner.profiles[name].eloP, "PEAK ELO", runner.profiles[name].peakElo)
        rank += 1
        if rank >= 21:
            break

    print("Serral's Match History", "EXPOVERALL:", runner.profiles['Serral'].expOverall, "WINPERCENTAGE:", runner.profiles['Serral'].wins / runner.profiles['Serral'].total)
    print("Maru's Match History", "EXPOVERALL:", runner.profiles['Maru'].expOverall, "WINPERCENTAGE:",
          runner.profiles['Maru'].wins / runner.profiles['Maru'].total)

    # USE FOR REAL NOW
    runner.createTTV(1.0,0.0)
    runner.updateModel()
    print("Used all matches for training, ready for deployment")
    maruAliveResults = runner.predict("Maru", "aLive")
    print("Maru", maruAliveResults[0], "aLive", maruAliveResults[1])
    print(runner.predictSeries("Maru", "aLive", 1))

    runner.runLive()