from __future__ import print_function
from PyAstronomy import pyasl

def convertSexaToDec(coord):
    """
    coord: This is an input string having the coordinates in sexagesimal convention
    """
    ra,dec = pyasl.coordsSexaToDeg(coord)
    return ra, dec
def convertDecToSexa(ra,dec):
    """
    ra: Radial Ascension in decimal convention
    dec:Declination in decimal convention
    """
    sexa = pyasl.coordsDegToSexa(ra,dec)
    return sexa