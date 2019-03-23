import unittest
import pytest
from player_profile import PlayerProfile

class GlickoTests(unittest.TestCase):
    def testGlicko(self):
        profile1 = PlayerProfile(name='name', race=None, firstDate="Saturday March 16 2019")
        profile1.glickoRating = 1500
        profile1.glickoRD = 200
        profile1.glickoVol = 0.06
        profile1.glickoTau = 0.5
        profile1.placementsLeft = 0

        # profile1.updateGlicko(opponentRating=1400, opponentRD=30, win=True)
        # profile1.updateGlicko(opponentRating=1550, opponentRD=100, win=False)
        # profile1.updateGlicko(opponentRating=1700, opponentRD=300, win=False)
        profile1.placementResults = [
            [1400, 30, True],
            [1550, 100, False],
            [1700, 300, False]
        ]
        profile1.placementsEnd()

        assert pytest.approx(profile1.glickoRating, 0.01) == 1464.06
        assert pytest.approx(profile1.glickoRD, 0.01) == 151.52
        assert pytest.approx(profile1.glickoVol, 0.0001) == 0.05999

if __name__ == '__main__':
    unittest.main()