import math
from datetime import datetime, timedelta


class PlayerProfile:
    def __init__(self, name, race, firstDate):
        self.name = name
        self.dateFormat = '%A %B %d %Y'
        self.lastPlayedDate = datetime.strptime(firstDate, self.dateFormat).date()
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
        self.glickoTau = 0.8
        self.placementsLeft = 15
        self.placementResults = []
        self.race = race

        # For glicko2 rating periods
        self.firstPlayedDate = datetime.strptime(firstDate, self.dateFormat).date()
        self.lastStartPeriod = datetime.strptime(firstDate, self.dateFormat).date()
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
        self.eloK = 32
        self.peakElo = 1200


    def updateProfile(self, date, opponentProfile, win):
        assert(type(win) == bool)
        self.total += 1
        winNum = 1 if win else 0
        self.wins += winNum

        date = datetime.strptime(date, self.dateFormat).date()
        playTimeGap = (date - self.lastPlayedDate).days // 30
        # NOTE: playTimeGap is in months
        self.lastPlayedDate = date

        if opponentProfile.race == 'Z':
            self.winsZ += 1 if win else 0
            self.totalZ += 1
            self.expZ = self.matchAlpha * winNum + self.expZ * (1 - self.matchAlpha)
        elif opponentProfile.race == 'T':
            self.winsT += 1 if win else 0
            self.totalT += 1
            self.expT = self.matchAlpha * winNum + self.expT * (1 - self.matchAlpha)
        elif opponentProfile.race == 'P':
            self.winsP += 1 if win else 0
            self.totalP += 1
            self.expP = self.matchAlpha * winNum + self.expP * (1 - self.matchAlpha)
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


        # ELO
        # if self.elo < 2100:
        #     self.eloK = 32
        # elif self.elo <= 2400:
        #     self.eloK = 24
        # else:
        #     self.eloK = 16
        if self.total < 30:
            self.eloK = 40
        elif self.elo < 2400 and self.peakElo < 2400:
            self.eloK = 20
        else:
            self.eloK = 10

        Q_A = 10**(self.elo/400)
        Q_B = 10**(opponentProfile.elo/400)
        E_A = Q_A / (Q_A + Q_B)

        # Update Elo
        self.elo = self.elo + self.eloK * (winNum - E_A)
        self.peakElo = max(self.elo, self.peakElo)

        self.expAverageLastPlayed = self.expAlpha * playTimeGap + self.expAverageLastPlayed * (1 - self.expAlpha)

    def checkDecay(self, date):
        numPeriods = (date - (self.lastStartPeriod + timedelta(days=self.periodDays))).days // self.periodDays
        if numPeriods > 0:
            self.lastStartPeriod += timedelta(days=self.periodDays * numPeriods)
            self.decay(numPeriods)

    def decay(self, periods):
        for i in range(periods):
            phi = self.glickoRD / 173.7178
            newPhi= math.sqrt(phi**2 + self.glickoVol**2)
            self.glickoRD = newPhi * 173.7178

    def updateGlicko(self, opponentRating, opponentRD, win):
        assert (type(win) == bool)
        winNum  = 1 if win else 0
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

            winNum = 1 if win else 0
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

    def getFeatures(self, opponentRace):
        mu = (self.glickoRating - 1500) / 173.7178
        phi = self.glickoRD / 173.7178
        if self.expAverageLastPlayed == 0:
            timeWeightedRating = self.glickoRating
        else:
            timeWeightedRating = self.glickoRating / self.expAverageLastPlayed

        if opponentRace == 'Z':
            normRace = self.expZ - self.expOverall
        elif opponentRace == 'T':
            normRace = self.expT - self.expOverall
        elif opponentRace == 'P':
            normRace = self.expP - self.expOverall
        else:
            normRace = 0
        return [mu, phi, timeWeightedRating, normRace, self.glickoRating, self.elo]

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

