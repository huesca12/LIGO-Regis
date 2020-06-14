#IMPORTS
#import external packages
import numpy as np
import pandas as pd

#import local modules
from analysis import pca
from analysis import tsne
from plot import pcaViz
from plot import tsneViz
from plot import pairplot

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
#TODO: update t-sne to run on all features
#TODO: look into optimization
#Very computationally intensive! Likely will not run on average laptop!
#runs T-SNE on the top three principal components and returns the results as columns of mainDf
#keep the following line commented out to not run tsne
#tsne.tsnePCA3(mainDf)

#DATA VISUALIZATION
#PCA
#visualizes the datatset using the top two principal components
pcaViz.viz(mainDf)

#T-TSNE
#visualizes the dataset using the t-sne results
#keep commented out if you haven't uncommented the t-sne section
#tsneViz.viz(mainDf)

#Pairplots
#TODO: Include functionality for an easier way to do an aggregate pairplot
#creates a pairplot for the specified lists of glitch(es) and features
glitches = ['Extremely_Loud','Blip']
features = ['centralFreq', 'duration', 'Q-value']
pairplot.pPlot(mainDf, glitches, features)

#Violin Plots
#TODO: Implement violin plots
#TODO: Include functionality for an easier way to do an aggregate violin plot
#creates violin plots to visualize feature distriubtion in a more ~novel~ way

print("Done!")
