import random
import threading
from datetime import datetime
from uuid import UUID

from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from gevent.pywsgi import WSGIServer
from wtforms import StringField, validators, SelectField, IntegerField

import helper
from model_runner import ModelRunner
from model_runner import NoBetsYetExceptions
from models.mlp import MLP


class PlayersForm(FlaskForm):
    player1 = StringField("Name of First Player (case insensitive)", [validators.DataRequired("Please enter a player name.")])
    player2 = StringField("Name of Second Player (case insensitive)", [validators.DataRequired("Please enter a player name.")])

class ChooseForm(FlaskForm):
    choice1 = SelectField('Player 1', validators=[validators.DataRequired("Please pick a player")])
    choice2 = SelectField('Player 2', validators=[validators.DataRequired("Please pick a player")])
    bo = IntegerField('Best Of', validators=[validators.DataRequired("Please enter maximum number of games in series")])
class API:
    def __init__(self, port=80, threaded=True, debug=True):
        self.port = port
        self.threaded=threaded
        self.debug=debug
        self.app = Flask(__name__.split('.')[0])
        self.app.config['SECRET_KEY'] = str(UUID(int=random.randint(0, 10000000000)))
        self.app.add_url_rule('/', 'index', self.index, methods=['GET','POST'])
        self.app.add_url_rule('/players/<p1>/<p2>', 'players', self.players, methods=['GET','POST'])
        self.app.add_url_rule('/bets', 'bets', self.bets)
        self.app.add_url_rule('/rankings', 'rankings', self.rankings)
        self.app.add_url_rule('/recent', 'recent', self.recentMatches)
        helper.fillRegionDict()
        model = MLP(max_epochs=10000, batch_size=128, learning_rate=5e-3, width=50, layers=1)
        print('Model Created')
        self.runner = ModelRunner(model, "data/matchResults_aligulac.csv", trainRatio=0.8, testRatio=0.2,
                             lastGameId="302054", keepPercent=1.0, decay=False)
        print(datetime.now(), 'Model Runner Created')
        self.runner.getLastId()
        self.runner.loadProfiles()
        self.runner.model.loadBackup()
        self.runner.clearMemory()
        print("Memory Cleared")

        rankingsList = self.runner.generateRanking(20)
        rank = 1
        print("Rank, Name, Race, Country, Project W/R, Elo, Glicko, MatchExpAvg")
        for [rate, profile] in rankingsList:
            print(rank, profile.name, profile.race, profile.country, rate, profile.elo, profile.glickoRating,
                  profile.expOverall)
            rank += 1

        self.liveThread = threading.Thread(target=self.runner.getLive)
        self.liveThread.start()

    def start(self):
        http_server = WSGIServer(('', self.port), self.app)
        http_server.serve_forever()
        # self.app.run(debug=self.debug, port=self.port, threaded=self.threaded)

    def index(self):
        form = PlayersForm()
        if form.validate_on_submit():
            return redirect(url_for('players', p1=form.player1.data, p2=form.player2.data))
        elif request.method == 'GET':
            return render_template('index.html', form=form)

    def players(self, p1, p2):

        p1List = self.runner.profiles[p1.lower()]
        p2List = self.runner.profiles[p2.lower()]

        firstKeyPairs = [(i, str(p1List[i])) for i in range(len(p1List))]
        secondKeyPairs = [(i, str(p2List[i])) for i in range(len(p2List))]

        cform = ChooseForm()
        cform.choice1.choices = firstKeyPairs
        cform.choice2.choices = secondKeyPairs
        if request.method == 'POST':
            bestOf = cform.bo.data
            profile1 = p1List[int(cform.choice1.data)]
            profile2 = p2List[int(cform.choice2.data)]

            seriesPred = self.runner.predictSeries(player1=profile1.name, p1Race=profile1.race,
                                                   p1Country=profile1.country, player2=profile2.name,
                                                   p2Race=profile2.race, p2Country=profile2.country, bestOf=bestOf)


            tableList = []
            tableList.append("<table><tr><th>Probability</th><th>Score</th><th>Score</th><th>Probability</th></tr>")
            keySize = len(seriesPred[0].keys())
            numKeys = 0
            t1 = 0
            t2 = 0
            for key in sorted(seriesPred[0].keys(), reverse=True):
                numKeys += 2
                seriesPred[0][key] = seriesPred[0][key].item()
                t1 += seriesPred[0][key]
                backwardsKey = key[::-1]
                seriesPred[0][backwardsKey] = seriesPred[0][backwardsKey].item()
                t2 += seriesPred[0][backwardsKey]
                tableList.append("<tr><td>")
                tableList.append("%.4f" % round(seriesPred[0][key], 4))
                tableList.append("</td>")
                tableList.append("<td>")
                tableList.append(key)
                tableList.append("</td>")
                tableList.append("<td>")
                tableList.append(backwardsKey)
                tableList.append("</td>")
                tableList.append("<td>")
                tableList.append("%.4f" % round(seriesPred[0][backwardsKey], 4))
                tableList.append("</td>")
                tableList.append("</tr>")
                if numKeys >= keySize:
                    break
            tableList.append("<tr><td>")
            tableList.append("%.4f" % round(t1, 4))
            tableList.append("</td>")
            tableList.append("<td>")
            tableList.append("Winner (")
            tableList.append(profile1.name)
            tableList.append(")</td><td>")
            tableList.append("Winner (")
            tableList.append(profile2.name)
            tableList.append(")</td>")
            tableList.append("<td>")
            tableList.append("%.4f" % round(t2, 4))
            tableList.append("</td></tr></table>")

            tableString = "".join(tableList)
            style = "<style>* {margin: 0;font-family: Arial, Helvetica, sans-serif;}table {font-family: arial, sans-serif;border-collapse: collapse;width: 100%;}td, th {border: 1px solid #dddddd;text-align: left;padding: 8px;}tr:nth-child(even) {background-color: #dddddd;}</style>"

            return "{6}<h2>Best of {4}: {0} {1} vs {3} {2}</h2><br><h3>{5}</h3>".format(profile1.name, "%.4f" % round(seriesPred[1].item(), 4), profile2.name, "%.4f" % round(seriesPred[2].item(), 4), bestOf, tableString, style)

        return render_template('players.html', form=cform)

    def bets(self):
        graph_url = ""
        try:
            graph_url = self.runner.graphBalance()
        except NoBetsYetExceptions:
            print("Not enough data points in balance history")
        return render_template("bets.html", graph=graph_url)

    def rankings(self):
        rankingsList = self.runner.generateRanking(20)
        ranks = []
        rates = []
        names = []
        races = []
        countries = []
        elos = []
        glickos = []

        rank = 1
        for [rate, profile] in rankingsList:
            ranks.append(rank)
            rates.append("%.4f" % round(rate, 4))
            names.append(profile.name)
            races.append(profile.race)
            countries.append(profile.country)
            elos.append("%.2f" % round(profile.elo, 2))
            glickos.append("%.2f" % round(profile.glickoRating, 2))
            rank += 1
        return render_template("rankings.html", ranks=ranks, rates=rates, names=names, races=races, countries=countries, elos=elos, glickos=glickos)

    def recentMatches(self):
        matchList = reversed(self.runner.recentMatches)
        return render_template("recent.html", matches=matchList)



if __name__ == "__main__":
    a = API()
    a.start()