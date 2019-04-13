import codecs
import csv
import random
import string
import time

import requests
from bs4 import BeautifulSoup

import helper


class Crawler:
    def __init__(self, url, fileName="../data/matchResults_regionsRaces.csv", cleaner=helper.gosuCleaner()):
        self.url = url
        self.matchList = []
        self.fieldNames = ['Date', 'Id', 'Player1', 'Player1Region', 'Player1Race', 'Player2', 'Player2Region',
                      'Player2Race', 'Score1', 'Score2']
        self.fileName = fileName
        self.cleaner = cleaner

    def start(self, fromPage=1):
        start = self.url
        page = requests.get(start).content
        soup = BeautifulSoup(page, features="html.parser")
        paginator = soup.select(".pagination > li > a")
        pages = int(paginator[-2]['aria-label'].split()[-1])

        # Get data from all pages in a parallel fashion
        # with Pool(processes=10) as pool:
        #     pool.map(self.para_getData, range(2, pages))
        if fromPage == 1:
            with codecs.open(self.fileName, "w", "utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldNames)
                writer.writeheader()

        for page in range(fromPage, pages + 1):
            self.para_getData(page)

        self.matchList.sort(key=lambda x: x.id)

        # with open("../data/matchResults.csv", "w") as file:
        #     for result in self.matchList:
        #         writeString = ",".join(result.getList()) + "\n"
        #         file.write(writeString)

    def para_getData(self, page):
        url = self.url + "&page={}"
        _ = self.getData(url.format(page))
        print(page, "done")

    def liveGenerator(self, fromPage=580, lastGameId=None):

        flag = False

        start = self.url
        page = requests.get(start).content
        soup = BeautifulSoup(page, features="html.parser")
        paginator = soup.select(".pagination > li > a")
        pages = int(paginator[-2]['aria-label'].split()[-1])
        assert(fromPage <= pages)
        nextPaged = False
        reachedLast = False
        print("GG Page:", fromPage)
        while not flag:
            url = start + "&page={}"

            gameDicts = self.getData(url.format(fromPage), writeFile=False)

            with codecs.open(self.fileName, "a", "utf-8") as file:

                writer = csv.DictWriter(file, fieldnames=self.fieldNames)

                tempLastId = lastGameId
                for gameDict in gameDicts:
                    # if nextPaged or lastGameId is None:
                    #     writer.writerow(gameDict)
                    #     yield gameDict
                    #     tempLastId = gameDict['Id']
                    if reachedLast or lastGameId is None:
                        writer.writerow(gameDict)
                        yield gameDict
                        tempLastId = gameDict['Id']
                    elif gameDict['Id'] == lastGameId:
                        reachedLast = True

                lastGameId = tempLastId

            if len(gameDicts) >= 18:
                paginator = soup.select(".pagination > li > a")
                pages = int(paginator[-2]['aria-label'].split()[-1])
                while fromPage >= pages:
                    time.sleep(60*60)
                    paginator = soup.select(".pagination > li > a")
                    pages = int(paginator[-2]['aria-label'].split()[-1])
                fromPage += 1
                nextPaged = True
                print("GG Page:", fromPage)
            else:
                nextPaged = False
                reachedLast = False
                randomizer = random.uniform(0.8, 1.2)
                time.sleep(60*60*2*randomizer)



    def getData(self, url, numTries=0, writeFile=True):
        soup = BeautifulSoup(requests.get(url).content, features="html.parser")
        cellMatches = soup.select("div.match.finished")

        localMatchList = []

        while len(cellMatches) == 0:
            randomizer = random.uniform(0.5, 1.5)
            if numTries > 13:
                time.sleep(10800*randomizer)
            elif numTries > 10:
                time.sleep(3600*randomizer)
            elif numTries > 8:
                time.sleep(300*randomizer)
            elif numTries > 5:
                time.sleep(30*randomizer)
            else:
                time.sleep(10*randomizer)
            numTries += 1
            soup = BeautifulSoup(requests.get(url).content, features="html.parser")
            cellMatches = soup.select("div.match.finished")

        for cell in cellMatches:
            id = cell["id"].lstrip("panel")
            player1_cells = cell.select("span.team-1")
            player2_cells = cell.select("span.team-2")

            assert(len(player1_cells) == 1 and len(player2_cells) == 1)

            player1 = self.cleaner.cleanName(player1_cells[0].get_text().strip())
            player2 = self.cleaner.cleanName(player2_cells[0].get_text().strip())

            results = cell.select("div.match-score > span")

            assert(len(results) == 3)

            result1 = results[0].get_text().strip()
            result2 = results[-1].get_text().strip()

            date = self.getDate(cell)


            matchPageURL = "https://www.gosugamers.net" + cell.a['href']
            matchPage = BeautifulSoup(requests.get(matchPageURL).content, features="html.parser")

            matchDataTries = 0
            while len(matchPage.select("div.game-data")) == 0:
                randomizer = random.uniform(0.5, 1.2)
                if matchDataTries > 13:
                    time.sleep(10800*randomizer)
                elif matchDataTries > 10:
                    time.sleep(3600*randomizer)
                elif matchDataTries > 8:
                    time.sleep(300*randomizer)
                elif matchDataTries > 5:
                    time.sleep(30*randomizer)
                else:
                    time.sleep(10*randomizer)
                matchDataTries += 1
                matchPage = BeautifulSoup(requests.get(matchPageURL).content, features="html.parser")

            try:
                region1 = matchPage.select("div.game-data > div.team-1 > div.row > div.region")[0].text.strip().translate(str.maketrans('', '', string.punctuation))
            except IndexError:
                region1 = ""
            try:
                region2 = matchPage.select("div.game-data > div.team-2 > div.row > div.region")[0].text.strip().translate(str.maketrans('', '', string.punctuation))
            except IndexError:
                region2 = ""
            try:
                race1 = matchPage.select("div.game-data > div.team-1 > div.row > span.faction")[0].text.strip().translate(str.maketrans('', '', string.punctuation))
            except IndexError:
                race1 = ""
            try:
                race2 = matchPage.select("div.game-data > div.team-2 > div.row > span.faction")[0].text.strip().translate(str.maketrans('', '', string.punctuation))
            except IndexError:
                race2 = ""

            self.matchList.append(matchResult(date=date, id=id, player1=player1, player2=player2, result1=result1,
                                              result2=result2, region1=region1, race1=race1, region2=region2, race2=race2))
            localMatchList.append(matchResult(date=date, id=id, player1=player1, player2=player2, result1=result1,
                                              result2=result2, region1=region1, race1=race1, region2=region2, race2=race2))
        print("matchlist length", len(self.matchList))
        outDicts = []
        if writeFile:
            with codecs.open(self.fileName, "a", "utf-8") as file:

                writer = csv.DictWriter(file, fieldnames=self.fieldNames)

                for result in localMatchList:
                    outDict = {'Date': result.date, 'Id': result.id, 'Player1': result.player1, 'Player1Region': result.region1,
                               'Player1Race': result.race1, 'Player2': result.player2, 'Player2Region': result.region2,
                               'Player2Race': result.race2, 'Score1': result.result1, 'Score2': result.result2}

                    writer.writerow(outDict)
                    outDicts.append(outDict)
        for result in localMatchList:
            outDict = {'Date': result.date, 'Id': result.id, 'Player1': result.player1, 'Player1Region': result.region1,
                       'Player1Race': result.race1, 'Player2': result.player2, 'Player2Region': result.region2,
                       'Player2Race': result.race2, 'Score1': result.result1, 'Score2': result.result2}
            outDicts.append(outDict)

        return outDicts
        # print(cellMatches)


    def getDate(self, soup):
        prev = soup.find_previous_sibling("div", class_="match-date")
        if prev is None:
            print("Could not find date")
            return "NaN"
        return prev.text.strip().replace(',', '')

class matchResult:
    def __init__(self, date, id, player1, player2, result1, result2, region1, race1, region2, race2):
        self.date = date
        self.id = id
        self.player1 = player1
        self.player2 = player2
        self.result1 = result1
        self.result2 = result2
        self.region1 = region1
        self.race1 = race1
        self.region2 = region2
        self.race2 = race2

    def getList(self):
        return [self.date, self.id, self.player1, self.player2, self.result1, self.result2]

def reverseFile(inFile, outFile):
    with codecs.open(inFile, "r", "utf-8") as inF, codecs.open(outFile, "w", "utf-8") as outF:
        csvR = csv.reader(inF)
        csvW = csv.writer(outF)
        csvW.writerow(next(csvR))
        csvW.writerows(reversed(list(csvR)))

if __name__ == '__main__':
    # reverseFile("../data/matchResultsDates.csv", "../data/matchResultsDates_reversed.csv")
    url = "https://www.gosugamers.net/starcraft2/matches/results?sortBy=date-asc&maxResults=18"
    newFileName = "../data/matchResults_aligulac.csv"
    c = Crawler(url, newFileName)
    c.start()

    #c = Crawler(url)
    # c.start()
    #liveGen = c.liveGenerator(fromPage=585, lastGameId="297304")
    #for i in liveGen:
    #    print(i)