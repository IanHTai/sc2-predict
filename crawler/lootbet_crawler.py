from bs4 import BeautifulSoup
import requests
import codecs
import datetime
import dateutil.parser
import dateutil.tz
import time
import random
import helper
import string

class LootCrawler:
    def __init__(self, url, gosuUrl, cleaner=helper.gosuCleaner()):
        self.url = url
        self.gosuUrl= gosuUrl
        self.cleaner = cleaner

    def getMatches(self):
        site = requests.get(self.url).content
        soup = BeautifulSoup(site, features="html.parser")
        matches = soup.select("app-match.match")
        if len(matches) == 0:
            return []
        # Apparently lootbet doesn't give datetime info to beautifulsoup?
        #timezone = soup.select("a.dropdown-time")[0].text.split()[-1]
        #timezone = dateutil.tz.tzoffset(datetime.datetime.now(dateutil.tz.tzlocal()).tzname())
        #timezone = str(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)
        #tzConvert = {"Pacific Daylight Time": "-07:00", "Pacific Standard Time": "-08:00"}

        #print(tzConvert[timezone])
        #timezone = tzConvert[timezone]
        timezone = "+00:00"
        lootMatches = []
        matchTries = 0

        while len(matches) == 0:
            randomizer = random.uniform(0.5, 1.2)
            if matchTries > 13:
                time.sleep(10800*randomizer)
            elif matchTries > 10:
                time.sleep(3600*randomizer)
            elif matchTries > 8:
                time.sleep(300*randomizer)
            elif matchTries > 5:
                time.sleep(30*randomizer)
            else:
                time.sleep(10*randomizer)
            matchTries += 1
            matchPage = BeautifulSoup(requests.get(self.url).content, features="html.parser")
            matches = matchPage.select("app-match.match")

        for match in matches:
            moneyLine = match.select("div.itemNew")[0]
            time_ = moneyLine.select("span.Date > span.time")[0].text.strip()
            date = moneyLine.select("span.Date > span.date")[0].text.strip()
            dateAndTime = time_ + " " + date + " " + timezone
            #dt = datetime.datetime.strptime(dateAndTime, "%H:%M %d %b %Z")
            dt = dateutil.parser.parse(dateAndTime)
            player1 = moneyLine.select("app-odd.teamLeft > span.name")[0].text.strip()
            player2 = moneyLine.select("app-odd.teamRight > span.name")[0].text.strip()

            #print(player1, player2)

            bestOf, id, race1, region1, race2, region2 = self.findBOInfo(dt, [player1, player2])
            if bestOf is None:
                continue
            odds1 = float(moneyLine.select("app-odd.teamLeft > span.cof")[0].text)
            odds2 = float(moneyLine.select("app-odd.teamRight > span.cof")[0].text)
            lootMatches.append(LootMatch(player1=player1, p1Race=race1, p1Country=region1, player2=player2,
                                         p2Race=race2, p2Country=region2, bestOf=bestOf, odds1=odds1, odds2=odds2, id=id, dt=dt))

        return lootMatches

    def findBOInfo(self, dt, players):
        site = requests.get(self.gosuUrl).content
        soup = BeautifulSoup(site, features="html.parser")
        matches = soup.select("div.cell.match.upcoming")
        matchTries = 0

        while len(matches) == 0:
            randomizer = random.uniform(0.5, 1.2)
            if matchTries > 13:
                time.sleep(10800*randomizer)
            elif matchTries > 10:
                time.sleep(3600*randomizer)
            elif matchTries > 8:
                time.sleep(300*randomizer)
            elif matchTries > 5:
                time.sleep(30*randomizer)
            else:
                time.sleep(10*randomizer)
            matchTries += 1
            matchPage = BeautifulSoup(requests.get(self.gosuUrl).content, features="html.parser")
            matches = matchPage.select("div.cell.match.upcoming")

        for match in matches:
            gosu_dt_str = match.select("span.post-date > time")[0].get("datetime")
            gosu_dt = dateutil.parser.parse(gosu_dt_str)
            id = match["id"].lstrip("panel")
            #print(gosu_dt, dt)

            if gosu_dt > dt + datetime.timedelta(minutes=15):
                return None, None, None, None, None, None

            if dt == gosu_dt or (gosu_dt - dt <= datetime.timedelta(minutes=15) and gosu_dt - dt >= datetime.timedelta(minutes=-15)):
                #print("time match")
                gosu_p1 = self.cleaner.cleanName(match.select("span.team-1")[0].text.strip())
                gosu_p2 = self.cleaner.cleanName(match.select("span.team-2")[0].text.strip())
                if (gosu_p1 == players[0] and gosu_p2 == players[1]) or (gosu_p1 == players[1] and gosu_p2 == players[0]):
                    matchPageURL = "https://www.gosugamers.net" + match.a['href']
                    matchPage = BeautifulSoup(requests.get(matchPageURL).content, features="html.parser")
                    matchDataTries = 0
                    if len(matchPage.select("div.best-of")) == 0:
                        if matchDataTries > 13:
                            time.sleep(10800)
                        elif matchDataTries > 10:
                            time.sleep(3600)
                        elif matchDataTries > 8:
                            time.sleep(300)
                        elif matchDataTries > 5:
                            time.sleep(30)
                        else:
                            time.sleep(1)
                        matchDataTries += 1
                        matchPage = BeautifulSoup(requests.get(matchPageURL).content, features="html.parser")
                    intArr = [int(s) for s in matchPage.select("div.best-of")[0].text.split() if s.isdigit()]

                    if gosu_p1 == players[0]:
                        try:
                            region1 = matchPage.select("div.game-data > div.team-1 > div.row > div.region")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            region1 = ""
                        try:
                            region2 = matchPage.select("div.game-data > div.team-2 > div.row > div.region")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            region2 = ""
                        try:
                            race1 = matchPage.select("div.game-data > div.team-1 > div.row > span.faction")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            race1 = ""
                        try:
                            race2 = matchPage.select("div.game-data > div.team-2 > div.row > span.faction")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            race2 = ""
                    else:
                        try:
                            region2 = matchPage.select("div.game-data > div.team-1 > div.row > div.region")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            region2 = ""
                        try:
                            region1 = matchPage.select("div.game-data > div.team-2 > div.row > div.region")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            region1 = ""
                        try:
                            race2 = matchPage.select("div.game-data > div.team-1 > div.row > span.faction")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            race2 = ""
                        try:
                            race1 = matchPage.select("div.game-data > div.team-2 > div.row > span.faction")[
                                0].text.strip().translate(str.maketrans('', '', string.punctuation))
                        except IndexError:
                            race1 = ""

                    if len(intArr) == 1:
                        return intArr[0], id, race1, region1, race2, region2
                    else:
                        return None, None, None, None, None, None

        return None, None

class LootMatch:
    def __init__(self, player1, p1Race, p1Country, player2, p2Race, p2Country, bestOf, odds1, odds2, id, dt):
        self.player1 = player1
        self.player2 = player2
        self.bestOf = bestOf
        self.odds1 = odds1
        self.odds2 = odds2
        self.id = id
        self.p1Race = p1Race
        self.p1Country = p1Country
        self.p2Race = p2Race
        self.p2Country = p2Country
        self.dt = dt

if __name__ == "__main__":
    lc = LootCrawler(url="https://loot.bet/sport/esports/starcraft",
                     gosuUrl="https://www.gosugamers.net/starcraft2/matches")
    matches = lc.getMatches()
    for match in matches:
        print(match.player1, match.odds1, match.bestOf, match.odds2, match.player2)