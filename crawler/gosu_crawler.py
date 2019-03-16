from bs4 import BeautifulSoup
import requests
from multiprocessing import Pool
import time
import codecs

class Crawler:
    def __init__(self, url):
        self.url = url
        self.matchList = []

    def start(self):
        start = self.url
        page = requests.get(start).content
        soup = BeautifulSoup(page, features="html.parser")
        #self.getData(soup)
        paginator = soup.select(".pagination > li > a")
        pages = int(paginator[-2]['aria-label'].split()[-1])

        # Get data from all pages in a parallel fashion
        # with Pool(processes=10) as pool:
        #     pool.map(self.para_getData, range(2, pages))

        for page in range(1, pages + 1):
            self.para_getData(page)

        self.matchList.sort(key=lambda x: x.id)

        # with open("../data/matchResults.csv", "w") as file:
        #     for result in self.matchList:
        #         writeString = ",".join(result.getList()) + "\n"
        #         file.write(writeString)

    def para_getData(self, page):
        url = self.url + "&page={}"
        soup = BeautifulSoup(requests.get(url.format(page)).content, features="html.parser")
        self.getData(soup)
        print(page, "done")

    def getData(self, soup, numTries=0):
        cellMatches = soup.select("div.match.finished")

        localMatchList = []

        if len(cellMatches) == 0:
            if numTries > 13:
                time.sleep(10800)
            elif numTries > 10:
                time.sleep(3600)
            elif numTries > 8:
                time.sleep(300)
            elif numTries > 5:
                time.sleep(30)
            else:
                time.sleep(1)
            return self.getData(soup, numTries + 1)

        for cell in cellMatches:
            id = cell["id"].lstrip("panel")
            player1_cells = cell.select("span.team-1")
            player2_cells = cell.select("span.team-2")

            assert(len(player1_cells) == 1 and len(player2_cells) == 1)

            player1 = player1_cells[0].get_text().strip()
            player2 = player2_cells[0].get_text().strip()

            results = cell.select("div.match-score > span")

            assert(len(results) == 3)

            result1 = results[0].get_text().strip()
            result2 = results[-1].get_text().strip()

            self.matchList.append(matchResult(id, player1, player2, result1, result2))
            localMatchList.append(matchResult(id, player1, player2, result1, result2))
        print("matchlist length", len(self.matchList))
        with codecs.open("../data/matchResults.csv", "a", "utf-8") as file:
            for result in localMatchList:
                writeString = ",".join(result.getList()) + "\n"
                file.write(writeString)

        # print(cellMatches)

class matchResult:
    def __init__(self, id, player1, player2, result1, result2):
        self.id = id
        self.player1 = player1
        self.player2 = player2
        self.result1 = result1
        self.result2 = result2

    def getList(self):
        return [self.id, self.player1, self.player2, self.result1, self.result2]

if __name__ == '__main__':
    url = "https://www.gosugamers.net/starcraft2/matches/results?maxResults=18"
    c = Crawler(url)
    c.start()
