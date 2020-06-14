#IMPORTS
#import external packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

#import local modules
from analysis import pca
from analysis import tsne

#DATA PREPARATION
#define the data filename
filename = 'data/gspy_o3a.csv' #your file here with a .csv ending

#read the csv file into a dataframe
rawDf = pd.read_csv(filename)

#define the list of columns that we want to drop
#note the inclusion of peakFreq and amplitude (original TSNE plot we have did not account for these)
dropList = ['chisq','chisqDof','confidence','GPStime','ifo','imgUrl','id']

#drop the columns we don't need
mainDf = rawDf.drop(columns=dropList)

#features only!
pcaDf = mainDf.drop(columns='label')

#DATA ANALYSIS
#PCA
#find the top three principal components and return them as columns of mainDf
pca.PCA3(pcaDf, mainDf)

#T-SNE
#Very computationally intensive! Likely will not run on average laptop!
#runs T-SNE on the top three principal components and returns the results as columns of mainDf
#keep the following line commented out to not run tsne
#tsne.tsnePCA3(mainDf)

print("Done!")
