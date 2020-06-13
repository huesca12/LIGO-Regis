#IMPORTS
#import external packages
import numpy as np
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

#import local modules

#
#define the data filename
filename = 'data/gspy_o3a.csv' #your file here with a .csv ending

#read the csv file into a dataframe
rawDf = pd.read_csv(filename)

#define the list of columns that we want to drop
dropList = ['chisq','chisqDof','confidence','GPStime','ifo','imgUrl','id']

#drop the columns we don't need
mainDf = rawDf.drop(columns=dropList)
