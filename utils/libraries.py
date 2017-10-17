import os
from os import listdir
from os.path import isfile, join
import sys
import psutil
import time
import timeit
import re
import math
import pandas as pd
import numpy as np
from glob import glob
import bcolz
import random
from imp import reload
import shutil
import argparse

import cv2 # python-opencv
from skimage import exposure # scikit-image
from scipy import misc

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import matplotlib.animation as mpl_animation
matplotlib.rc('animation', html='html5')

from IPython.display import display 
from PIL import Image # Pillow

import scipy
from scipy.misc.pilutil import imread, imresize

import sklearn
from sklearn.metrics import *

import keras
from keras import backend as K
from keras import utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.optimizers import *
from keras.losses import *
from keras.callbacks import *
from keras.utils.np_utils import to_categorical

# data preprocessing functions
#from keras.applications.imagenet_utils import preprocess_input 
#from keras.applications.inception_v3 import preprocess_input as preprocess_input_iv3
#from keras.applications.xception import preprocess_input as preprocess_input_x
#from keras.applications.mobilenet import preprocess_input as preprocess_input_mn

from tensorflow.python.client import device_lib

import utils_callbacks
from utils_callbacks import *

import metrics
from metrics import * 

from sklearn.model_selection import StratifiedKFold