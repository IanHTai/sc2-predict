import math
class PlayerProfile:
    def __init__(self, name, race, firstDate):
        self.lastPlayedDate = firstDate
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
        self.glickoTau = 0.7
        self.placementsLeft = 10
        self.placementResults = []
        self.race = None

        self.expAverageLastPlayed = 0
        self.expAlpha = 0.2

    def updateProfile(self, date, opponentProfile, win):
        self.total += 1
        self.wins += 1 if win else 0

        if opponentProfile.race == 'Z':
            self.winsZ += 1 if win else 0
            self.totalZ += 1
        elif opponentProfile.race == 'T':
            self.winsT += 1 if win else 0
            self.totalT += 1
        elif opponentProfile.race == 'P':
            self.winsP += 1 if win else 0
            self.totalP += 1

        if self.placementsLeft > 0:
            self.placementResults.append([opponentProfile.glickoRating, opponentProfile.glickoRD, win])
            if self.placementsLeft == 1:
                self.placementsEnd()
            self.placementsLeft -= 1
        else:
            self.updateGlicko(opponentProfile.glickoRating, opponentProfile.glickoRD, win)

        playTimeGap = date - self.lastPlayedDate
        self.expAverageLastPlayed = self.expAlpha * playTimeGap + self.expAverageLastPlayed * (1 - self.expAlpha)
        

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

        phiStar = math.sqrt(phi ** 2 + (sigmaPrime ** 2))

        phiPrime = 1 / math.sqrt((1 / (phiStar ** 2)) + (1 / v))
        muPrime = mu + phiPrime ** 2 * preMultiDelta

        rPrime = 173.7178 * muPrime + 1500
        RDPrime = 173.7178 * phiPrime

        self.glickoRating = rPrime
        self.glickoRD = RDPrime

    def updateGlicko(self, opponentRating, opponentRD, win):
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


    def glickoG(self, phi):
        return 1 / (math.sqrt(1 + (3 * (phi**2) / (math.pi ** 2))))

    def glickoExpected(self, muSelf, muOpponent, phi):
        return 1 / (1 + math.exp(-self.glickoG(phi) * (muSelf - muOpponent)))

    def getFeatures(self):
        return None