"""
Example python file creating GUI using Dans_Diffraction
"""

import sys,os
#import matplotlib
#matplotlib.use('tkagg')
#import numpy as np
#import matplotlib.pyplot as plt # Plotting


cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


dif.Startgui()
#xtl = dif.Crystal()

