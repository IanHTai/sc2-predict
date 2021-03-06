import math
from datetime import datetime, timedelta

import helper


class PlayerProfile:
    def __init__(self, name, race, firstDate=None, region=""):
        self.name = name
        self.dateFormat = '%A %B %d %Y'
        if not firstDate is None:
            self.lastPlayedDate = datetime.strptime(firstDate, self.dateFormat).date()
            # For glicko2 rating periods
            self.firstPlayedDate = datetime.strptime(firstDate, self.dateFormat).date()
            self.lastStartPeriod = datetime.strptime(firstDate, self.dateFormat).date()
        else:
            self.lastPlayedDate = None
            self.firstPlayedDate = None
            self.lastStartPeriod = None

        self.wins = 0
        self.winsZ = 0
        self.winsT = 0
        self.winsP = 0
        self.total = 0
        self.totalZ = 0
        self.totalT = 0
        self.totalP = 0
        self.glickoRating = 1500
        self.glickoRD = 350
        self.glickoVol = 0.06
        self.glickoTau = 0.3
        self.placementsLeft = 20
        self.placementResults = []
        self.race = race
        self.eloDecay = 0.05

        if region in helper.REGION_DICT.keys():
            self.region = helper.REGION_DICT[region]
        else:
            self.region = "OTHER"
        self.country = region

        # For glicko2 rating periods
        self.periodDays = 40

        self.expAverageLastPlayed = 0
        self.expAlpha = 0.2

        self.matchAlpha = 0.05
        self.expZ = 0.5
        self.expT = 0.5
        self.expP = 0.5
        self.expOverall = 0.5

        # ELO
        self.elo = 1200
        self.eloZ = 1200
        self.eloT = 1200
        self.eloP = 1200

        self.eloK = 30
        self.peakElo = 1200
        self.peakEloZ = 1200
        self.peakEloT = 1200
        self.peakEloP = 1200

        self.head2headExpDict = {}
        self.head2headExpAlpha = 0.2

    def __str__(self):
        return "{}, {}, {} -- Elo: {}".format(self.name, self.race, self.country, self.elo)

    def updateRace(self, race):
        self.race = race

    def updateProfile(self, date, opponentProfile, win):
        #date = datetime.strptime(date, self.dateFormat).date()

        if self.lastPlayedDate is None:
            self.lastPlayedDate = date
        if self.firstPlayedDate is None:
            self.firstPlayedDate = date
        if self.lastStartPeriod is None:
            self.lastStartPeriod = date

        self.total += 1
        winNum = win
        self.wins += winNum

        playTimeGap = (date - self.lastPlayedDate).days // 30
        # NOTE: playTimeGap is in months
        self.lastPlayedDate = date

        if opponentProfile.race == "Zerg":
            self.winsZ += win
            self.totalZ += 1
            self.expZ = self.matchAlpha * winNum + self.expZ * (1 - self.matchAlpha)
            if self.totalZ < 10:
                self.eloK = 30
            elif self.eloZ < 2400 and self.peakEloZ < 2400:
                self.eloK = 15
            else:
                self.eloK = 10
            if not self.region == opponentProfile.region:
                self.eloK *= 2

            Q_A = 10 ** (self.eloZ / 400)
            Q_B = 10 ** ((opponentProfile.eloZ + opponentProfile.eloT + opponentProfile.eloP) / 3. / 400)
            if self.race == "Zerg":
                Q_B = 10 ** (opponentProfile.eloZ / 400)
            elif self.race == "Terran":
                Q_B = 10 ** (opponentProfile.eloT / 400)
            elif self.race == "Protoss":
                Q_B = 10 ** (opponentProfile.eloP / 400)
            E_A = Q_A / (Q_A + Q_B)

            # Update Elo
            self.eloZ = self.eloZ + self.eloK * (winNum - E_A)
            self.peakEloZ = max(self.eloZ, self.peakEloZ)

        elif opponentProfile.race == "Terran":
            self.winsT += win
            self.totalT += 1
            self.expT = self.matchAlpha * winNum + self.expT * (1 - self.matchAlpha)
            if self.totalT < 10:
                self.eloK = 30
            elif self.eloT < 2400 and self.peakEloT < 2400:
                self.eloK = 15
            else:
                self.eloK = 10
            if not self.region == opponentProfile.region:
                self.eloK *= 2

            Q_A = 10 ** (self.eloT / 400)
            Q_B = 10 ** ((opponentProfile.eloZ + opponentProfile.eloT + opponentProfile.eloP) / 3. / 400)
            if self.race == "Zerg":
                Q_B = 10 ** (opponentProfile.eloZ / 400)
            elif self.race == "Terran":
                Q_B = 10 ** (opponentProfile.eloT / 400)
            elif self.race == "Protoss":
                Q_B = 10 ** (opponentProfile.eloP / 400)
            E_A = Q_A / (Q_A + Q_B)

            # Update Elo
            self.eloT = self.eloT + self.eloK * (winNum - E_A)
            self.peakEloT = max(self.eloT, self.peakEloT)

        elif opponentProfile.race == "Protoss":
            self.winsP += win
            self.totalP += 1
            self.expP = self.matchAlpha * winNum + self.expP * (1 - self.matchAlpha)
            if self.totalP < 10:
                self.eloK = 30
            elif self.eloP < 2400 and self.peakEloP < 2400:
                self.eloK = 15
            else:
                self.eloK = 10
            if not self.region == opponentProfile.region:
                self.eloK *= 2

            Q_A = 10 ** (self.eloZ / 400)
            Q_B = 10 ** ((opponentProfile.eloZ + opponentProfile.eloT + opponentProfile.eloP) / 3. / 400)
            if self.race == "Zerg":
                Q_B = 10 ** (opponentProfile.eloZ / 400)
            elif self.race == "Terran":
                Q_B = 10 ** (opponentProfile.eloT / 400)
            elif self.race == "Protoss":
                Q_B = 10 ** (opponentProfile.eloP / 400)
            E_A = Q_A / (Q_A + Q_B)

            # Update Elo
            self.eloP = self.eloP + self.eloK * (winNum - E_A)
            self.peakEloP = max(self.eloP, self.peakEloP)

        self.expOverall = self.matchAlpha * winNum + self.expOverall * (1 - self.matchAlpha)

        if self.placementsLeft > 0:
            self.placementResults.append([opponentProfile.glickoRating, opponentProfile.glickoRD, win])
            if self.placementsLeft == 1:
                self.placementsEnd()
                self.lastStartPeriod = date
            self.placementsLeft -= 1
        elif (self.lastPlayedDate - self.lastStartPeriod).days == self.periodDays:
            self.placementResults.append([opponentProfile.glickoRating, opponentProfile.glickoRD, win])
            self.placementsEnd()
            self.lastStartPeriod = date
        elif (self.lastPlayedDate - self.lastStartPeriod).days > self.periodDays:
            self.placementsEnd()
            self.placementResults = [[opponentProfile.glickoRating, opponentProfile.glickoRD, win]]
            self.checkDecay(date)
            self.lastStartPeriod = date
            #self.updateGlicko(opponentProfile.glickoRating, opponentProfile.glickoRD, win)
        else:
            self.placementResults.append([opponentProfile.glickoRating, opponentProfile.glickoRD, win])

        # Update general Elo
        if self.total < 30:
            self.eloK = 30
        elif self.elo < 2400 and self.peakElo < 2400:
            self.eloK = 15
        else:
            self.eloK = 10
        if not self.region == opponentProfile.region:
            self.eloK *= 2

        Q_A = 10 ** (self.elo / 400)
        Q_B = 10 ** (opponentProfile.elo / 400)
        E_A = Q_A / (Q_A + Q_B)

        # Update Elo
        self.elo = self.elo + self.eloK * (winNum - E_A)
        self.peakElo = max(self.elo, self.peakElo)



        self.expAverageLastPlayed = self.expAlpha * playTimeGap + self.expAverageLastPlayed * (1 - self.expAlpha)


        # Update head2head
        encodedName = self.head2headEncode(opponentProfile)
        if not encodedName in self.head2headExpDict:
            self.head2headExpDict[encodedName] = 0.5
        self.head2headExpDict[encodedName] = self.head2headExpAlpha * winNum + (1 - self.head2headExpAlpha) * self.head2headExpDict[encodedName]

    def head2headEncode(self, profile):
        if isinstance(profile.name, float) or isinstance(profile.race, float) or isinstance(profile.country, float):
            print(profile.name, profile.race, profile.country)
        return "".join((profile.name, profile.race, profile.country))


    def checkDecay(self, date):
        if self.lastStartPeriod is None:
            return

        # Decays both glicko and elo
        numPeriods = (date - (self.lastStartPeriod + timedelta(days=self.periodDays))).days // self.periodDays
        if numPeriods > 0:
            self.lastStartPeriod += timedelta(days=self.periodDays * numPeriods)
            self.decay(numPeriods)
            self.decayElo(numPeriods)
            self.decayHead2Head(numPeriods)
        if self.lastStartPeriod - date >= timedelta(days=self.periodDays):
            self.lastStartPeriod = date

    def decay(self, periods):
        for i in range(periods):
            phi = self.glickoRD / 173.7178
            newPhi= math.sqrt(phi**2 + self.glickoVol**2)
            self.glickoRD = newPhi * 173.7178

    def decayElo(self, periods):
        self.elo = (1-self.eloDecay)**periods * (self.elo - 1200) + 1200
        self.eloZ = (1-self.eloDecay)**periods * (self.eloZ - 1200) + 1200
        self.eloT = (1 - self.eloDecay) ** periods * (self.eloT - 1200) + 1200
        self.eloP = (1 - self.eloDecay) ** periods * (self.eloP - 1200) + 1200

    def decayHead2Head(self, periods):
        for encodedName in self.head2headExpDict:
            for i in range(0, periods):
                self.head2headExpDict[encodedName] = self.head2headExpAlpha * 0.5 + (1 - self.head2headExpAlpha) * self.head2headExpDict[encodedName]

    def updateGlicko(self, opponentRating, opponentRD, win):
        winNum  = win
        mu = (self.glickoRating - 1500) / 173.7178
        phi = self.glickoRD / 173.7178
        muOpponent = (opponentRating - 1500) / 173.7178
        phiOpponent = opponentRD / 173.7178

        expected = self.glickoExpected(mu, muOpponent, phiOpponent)
        v = (self.glickoG(phiOpponent) ** 2 * expected * (1 - expected)) ** -1
        preMultiDelta = self.glickoG(phiOpponent) * (winNum - expected)
        delta = preMultiDelta * v

        sigmaPrime = self.glickoNewVol(self.glickoVol, delta, phi, v, self.glickoTau)

        phiStar = math.sqrt(phi ** 2 + (sigmaPrime ** 2))

        phiPrime = 1 / math.sqrt((1 / (phiStar ** 2)) + (1 / v))
        muPrime = mu + phiPrime ** 2 * preMultiDelta

        rPrime = 173.7178 * muPrime + 1500
        RDPrime = 173.7178 * phiPrime

        self.glickoRating = rPrime
        self.glickoRD = RDPrime

    def placementsEnd(self):
        # GLICKO2, info found here http://www.glicko.net/glicko/glicko2.pdf

        mu = (self.glickoRating - 1500) / 173.7178
        phi = self.glickoRD / 173.7178
        oneOverV = 0
        preMultiDelta = 0
        for [opponentRating, opponentRD, win] in self.placementResults:
            muOpponent = (opponentRating - 1500) / 173.7178
            phiOpponent = opponentRD / 173.7178
            expected = self.glickoExpected(mu, muOpponent, phiOpponent)
            oneOverV += (self.glickoG(phiOpponent) ** 2) * expected * (1 - expected)

            winNum = win
            preMultiDelta += self.glickoG(phiOpponent) * (winNum - expected)
        v = oneOverV ** -1
        delta = preMultiDelta * v

        sigmaPrime = self.glickoNewVol(self.glickoVol, delta, phi, v, self.glickoTau)
        self.glickoVol = sigmaPrime

        phiStar = math.sqrt(phi ** 2 + (sigmaPrime ** 2))

        phiPrime = 1 / math.sqrt((1 / (phiStar ** 2)) + (1 / v))
        muPrime = mu + phiPrime ** 2 * preMultiDelta

        rPrime = 173.7178 * muPrime + 1500
        RDPrime = 173.7178 * phiPrime

        self.glickoRating = rPrime
        self.glickoRD = RDPrime

    def getFeatures(self, opponentProfile, useRaceRatio=False, useRD=True):
        mu = (self.glickoRating - 1500) / 173.7178
        phi = self.glickoRD / 173.7178
        if self.expAverageLastPlayed <= 1:
            timeWeightedRating = self.glickoRating
        else:
            timeWeightedRating = self.glickoRating / self.expAverageLastPlayed

        raceElo = self.elo
        raceQ_A = 10 ** (self.elo / 400)
        raceQ_B = 10 ** (opponentProfile.elo / 400)
        raceEXP = self.expOverall
        raceEloRatio = 1./3.
        if opponentProfile.race == "Zerg":
            normRace = self.expZ - self.expOverall
            raceElo = self.eloZ
            raceQ_A = 10 ** (self.eloZ / 400)
            raceEXP = self.expZ
            raceEloRatio = self.eloZ / (self.eloZ + self.eloT + self.eloP)
        elif opponentProfile.race == "Terran":
            normRace = self.expT - self.expOverall
            raceElo = self.eloT
            raceQ_A = 10 ** (self.eloT / 400)
            raceEXP = self.expT
            raceEloRatio = self.eloT / (self.eloZ + self.eloT + self.eloP)
        elif opponentProfile.race == "Protoss":
            normRace = self.expP - self.expOverall
            raceElo = self.eloP
            raceQ_A = 10 ** (self.eloP / 400)
            raceEXP = self.expP
            raceEloRatio = self.eloP / (self.eloZ + self.eloT + self.eloP)
        else:
            normRace = 0.5

        if self.race == "Zerg":
            raceQ_B = 10 ** (opponentProfile.eloZ / 400)
        elif self.race == "Terran":
            raceQ_B = 10 ** (opponentProfile.eloT / 400)
        elif self.race == "Protoss":
            raceQ_B = 10 ** (opponentProfile.eloP / 400)
        raceE_A = raceQ_A / (raceQ_A + raceQ_B)

        opponentMu = (opponentProfile.glickoRating - 1500) / 173.7178
        opponentPhi = opponentProfile.glickoRD / 173.7178
        glickoE = self.glickoExpected(mu, opponentMu, opponentPhi)

        Q_A = 10 ** (self.elo / 400)
        Q_B = 10 ** (opponentProfile.elo / 400)
        E_A = Q_A / (Q_A + Q_B)

        h2hExp = 0.5
        encodedName = self.head2headEncode(opponentProfile)
        if encodedName in self.head2headExpDict:
            h2hExp = self.head2headExpDict[encodedName]

        # outArr = [mu, phi, timeWeightedRating, normRace, glickoE, self.elo, E_A, raceElo, raceE_A, self.expOverall, raceEXP, h2hExp]
        # Raw Elo seems to have a very negative coeff (-0.5+), so I'm removing it as a feature
        outArr = [mu, timeWeightedRating, normRace, glickoE, E_A, raceElo, raceE_A, self.expOverall,
                  raceEXP, h2hExp, self.expAverageLastPlayed]
        if useRaceRatio:
            outArr.append(raceEloRatio)
            # raceEloRatio seems to be noisy
        if useRD:
            outArr.append(phi)

        return outArr

    """
    Helper functions for Glicko calculations
    """
    def glickoNewVol(self, sigma, delta, phi, v, tau):
        a = math.log(sigma ** 2)
        A = math.log(sigma ** 2)
        epsilon = 0.000001
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while self.glickoF(x=(a - k * tau), delta=delta, phi=phi, v=v, a=a, tau=tau) < 0:
                k += 1
            B = a - k * tau

        fA = self.glickoF(x=A, delta=delta, phi=phi, v=v, a=a, tau=tau)
        fB = self.glickoF(x=B, delta=delta, phi=phi, v=v, a=a, tau=tau)

        while abs(B - A) > epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = self.glickoF(x=C, delta=delta, phi=phi, v=v, a=a, tau=tau)
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2
            B = C
            fB = fC

        return math.exp(A / 2)


    def glickoF(self, x, delta, phi, v, a, tau):
        left = (math.e**x * (delta ** 2 - phi ** 2 - v - math.e**x)) / (2 * ((phi ** 2 + v + math.e**x) ** 2))
        right = (x - a) / (tau**2)
        return left - right

    def glickoG(self, phi):
        return 1 / (math.sqrt(1 + (3 * (phi**2) / (math.pi ** 2))))

    def glickoExpected(self, muSelf, muOpponent, phi):
        return 1 / (1 + math.exp(-self.glickoG(phi) * (muSelf - muOpponent)))

    """
    End of Glicko functions
    """

