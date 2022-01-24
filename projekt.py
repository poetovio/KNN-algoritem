import numpy as np
import math

#funkcija za izračun evklidske razdalje

def evklid(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y1 - y2)**2)
