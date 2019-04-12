import psycopg2
import pandas as pd
import config_sql as creds
import pycountry
import datetime

class AligulacSQL:
    """
    Convert SQL dump to database to standard csv file format
    db from http://aligulac.com/about/db/
    """
    def __init__(self):
        self.fieldNames = ['Date', 'Id', 'Player1', 'Player1Region', 'Player1Race', 'Player2', 'Player2Region',
                           'Player2Race', 'Score1', 'Score2']
        conn_string = "host=" + creds.PGHOST + " port=" + "5432" + " dbname=" + creds.PGDATABASE + " user=" + creds.PGUSER \
                      + " password=" + creds.PGPASSWORD
        self.dbConn = psycopg2.connect(conn_string)
        self.cursor = self.dbConn.cursor()

    def loadAllMatches(self):
        command = "SELECT MATCH.date, MATCH.id, p1.tag as t1, p1.country as c1, MATCH.rca as r1, p2.tag as t2, p2.country as c2, MATCH.rcb as r2, MATCH.sca, MATCH.scb FROM MATCH INNER JOIN PLAYER as p1 ON p1.id = MATCH.pla_id INNER JOIN PLAYER as p2 ON p2.id = MATCH.plb_id ORDER BY date ASC;"
        data = pd.read_sql(command, self.dbConn)
        print(data.shape)
        return data

    def countryMap(self, code):
        if code is None:
            return ""
        elif code == "UK":
            return "United Kingdom"
        elif code == "VN":
            return "Vietnam"
        else:
            if pycountry.countries.get(alpha_2=code) is None:
                print(code)
            return pycountry.countries.get(alpha_2=code).name.replace(',','')

    def raceMap(self, raceShort):
        if raceShort == "Z":
            return "Zerg"
        if raceShort == "T":
            return "Terran"
        if raceShort == "P":
            return "Protoss"
        if raceShort == "R":
            return "Random"
        return ""

    def dateMap(self, date):
        return "{dt:%A} {dt:%B} {dt.day} {dt:%Y}".format(dt=date)

    def standardizeData(self, df):
        columnRename = {'date':'Date', 'id':'Id', 't1':'Player1', 'c1':'Player1Region', 'r1':'Player1Race', 't2':'Player2',
                        'c2':'Player2Region', 'r2':'Player2Race', 'sca':'Score1', 'scb':'Score2'}
        df.rename(columns=columnRename, inplace=True)

        df['Player1Region'] = list(map(self.countryMap, df['Player1Region']))
        df['Player2Region'] = list(map(self.countryMap, df['Player2Region']))
        df['Player1Race'] = list(map(self.raceMap, df['Player1Race']))
        df['Player2Race'] = list(map(self.raceMap, df['Player2Race']))
        df['Date'] = list(map(self.dateMap, df['Date']))
        return df

    def loadToFile(self, fileName):
        df = self.standardizeData(self.loadAllMatches())
        df.to_csv(path_or_buf=fileName, index=False, encoding='utf-8')

if __name__ == '__main__':
    a = AligulacSQL()

    a.loadToFile("../data/matchResults_aligulac.csv")