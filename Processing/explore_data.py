###script to view and almagamate eye tracking files.
import numpy as np
import sys, os
from file_methods import *
import pickle
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as pd
import math as mt 
from scipy.interpolate import interp1d
from nslr_hmm import *
from timeit import default_timer as timer

from pathlib import Path
from pprint import pprint
import feather



steergaze_df = pd.read_feather("../Post-processing/trout_rerun.feather")

for group, data in steergaze_df.groupby(['trialcode']):

    print(group)
    plt.plot(data.midline_ref_onscreen_x.values, data.midline_ref_onscreen_z.values, 'o', alpha = .2)
    plt.show()
    
