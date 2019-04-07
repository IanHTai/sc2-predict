from bs4 import BeautifulSoup
import requests
import codecs
import datetime
import dateutil.parser
import dateutil.tz
import time

class LootCrawler:
    def __init__(self, url, gosuUrl):
        self.url = url
        self.gosuUrl= gosuUrl

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
        if len(matches) == 0:
            if matchTries > 13:
                time.sleep(10800)
            elif matchTries > 10:
                time.sleep(3600)
            elif matchTries > 8:
                time.sleep(300)
            elif matchTries > 5:
                time.sleep(30)
            else:
                time.sleep(1)
                matchTries += 1
            matchPage = BeautifulSoup(requests.get(self.url).content, features="html.parser")

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

            bestOf, id = self.findBOInfo(dt, [player1, player2])
            if bestOf is None:
                continue
            odds1 = float(moneyLine.select("app-odd.teamLeft > span.cof")[0].text)
            odds2 = float(moneyLine.select("app-odd.teamRight > span.cof")[0].text)
            lootMatches.append(LootMatch(player1=player1, player2=player2,bestOf=bestOf,odds1=odds1,odds2=odds2,id=id))

        return lootMatches

    def findBOInfo(self, dt, players):
        site = requests.get(self.gosuUrl).content
        soup = BeautifulSoup(site, features="html.parser")
        matches = soup.select("div.cell.match.upcoming")
        matchTries = 0

        if len(matches) == 0:
            if matchTries > 13:
                time.sleep(10800)
            elif matchTries > 10:
                time.sleep(3600)
            elif matchTries > 8:
                time.sleep(300)
            elif matchTries > 5:
                time.sleep(30)
            else:
                time.sleep(1)
                matchTries += 1
            matchPage = BeautifulSoup(requests.get(self.gosuUrl).content, features="html.parser")

        for match in matches:
            gosu_dt_str = match.select("span.post-date > time")[0].get("datetime")
            gosu_dt = dateutil.parser.parse(gosu_dt_str)
            id = match["id"].lstrip("panel")
            #print(gosu_dt, dt)

            if gosu_dt > dt + datetime.timedelta(minutes=15):
                return None, None

            if dt == gosu_dt or (gosu_dt - dt <= datetime.timedelta(minutes=15) and gosu_dt - dt >= datetime.timedelta(minutes=-15)):
                #print("time match")
                if match.select("span.team-1")[0].text.strip() == players[0] or match.select("span.team-1")[0].text.strip() == players[1]:
                    #print("player1 match")
                    if match.select("span.team-2")[0].text.strip() == players[0] or match.select("span.team-2")[0].text.strip() == players[1]:
                        #print("player2 match")
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
                        if len(intArr) == 1:
                            return intArr[0], id
                        else:
                            return None, None

        return None, None

class LootMatch:
    def __init__(self, player1, player2, bestOf, odds1, odds2, id):
        self.player1 = player1
        self.player2 = player2
        self.bestOf = bestOf
        self.odds1 = odds1
        self.odds2 = odds2
        self.id = id

if __name__ == "__main__":
    lc = LootCrawler(url="https://loot.bet/sport/esports/starcraft",
                     gosuUrl="https://www.gosugamers.net/starcraft2/matches")
    matches = lc.getMatches()
    for match in matches:
        print(match.player1, match.odds1, match.bestOf, match.odds2, match.player2)