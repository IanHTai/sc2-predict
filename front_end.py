from flask import Flask, render_template, request, redirect, url_for
from model_runner import ModelRunner
from models.mlp import MLP
from gevent.pywsgi import WSGIServer
from datetime import datetime
import helper
from flask_wtf import FlaskForm
from wtforms import StringField, validators, SelectField, IntegerField
from uuid import UUID
import random

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
        # self.app.add_url_rule('/inferenceResult?p1=<player1>&p1R=<player1Race>&p1C=<player1Country>&p2=<player2>&p2R=<player2Race>&p2C=<player2Country>&bo=<bestOf>', 'inferenceResult', self.inferenceResult)
        helper.fillRegionDict()
        model = MLP(max_epochs=10000, batch_size=128, learning_rate=5e-3)
        print('Model Created')
        self.runner = ModelRunner(model, "data/matchResults_aligulac.csv", trainRatio=0.8, testRatio=0.2,
                             lastGameId="296378", keepPercent=1.0)
        print(datetime.now(), 'Model Runner Created')
        self.runner.runFile(keepFeats=False, skipProb=0.8)
        print(datetime.now(), 'File Run')
        self.runner.model.loadBackup()
        maruAliveResults = self.runner.predict("maru", "Terran", "Korea Republic of", "alive", "Terran", "Korea Republic of")
        print("Maru", maruAliveResults[0], "aLive", maruAliveResults[1])
        print(self.runner.predictSeries("maru", "Terran", "Korea Republic of", "alive", "Terran", "Korea Republic of", 1))

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

            return "<h2>Best of {4}: {0} {1} vs {3} {2}</h2>".format(profile1.name, seriesPred[1], profile2.name, seriesPred[2], bestOf)

        return render_template('players.html', form=cform)


    # def postName(self, name):

if __name__ == "__main__":
    a = API()
    a.start()