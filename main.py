# IMPORTS
# import external packages
# import numpy as np
import pandas as pd

# import local modules
from analysis import pca
from analysis import tsne
from plot import pcaViz
from plot import tsneViz
from plot import pairplot
from simplify import simplify

# DATA PREPARATION
# ask if the user wants to use gspy_o3a or gspy_o3b
query = input("Would you like to analyze the o3a or o3b data set? (a/b): ")
# make sure the user inputted either "a" or "b"
assert query == "a" or query == "b", "You must input either \"a\" or \"b\""
# define file path to particular dataset
filename = f"data/gspy_o3{query}.csv"

# ask if the user wants to run the simplify pipeline
query = input("Would you like to run the simplify data preparation pipeline? (y/n): ")

# test if the user wants to run simplify
if query == "y":
    # if so, also ask for the desired sample number (and convert str to int)
    n = int(input("Please enter the desired sample size: "))
    # asks if a file should be saved
    query = input("Woud you like to save the simplified data set to a file? (y/n): ")
    if query == "y":
        # if so, set save_to_file parameter to True
        save = True
        # also, ask for an output filename
        simplified_filename = input("Output CSV filename: ")
    else:
        # otherwise, save_to_file parameter = None
        save = False
        # and we need no file name (equating this to None is implicitly tells simplify_csv to not process it)
        simplified_filename = None
    # get the df output from simplify and store it in mainDf
    rawDf = simplify.simplify_csv(filename, int(n), csv_out=simplified_filename, save_to_file=save)
# data preparation if simplify pipeline is not run
else:
    # read the csv file into a dataframe
    rawDf = pd.read_csv(filename)

# define the list of columns that we want to drop
# note the inclusion of peakFreq and amplitude (original TSNE plot we have did not account for these)
dropList = ["chisq", "chisqDof", "GPStime", "ifo", "imgUrl", "id"]
# drop the columns we don"t need
mainDf = rawDf.drop(columns=dropList)

# ask if the user wants a minimum confidence rating
query = input("Would you like to drop data entries below a particular GSpy confidence rating? (y/n): ")
if query == "y":
    # if so, ask for the desired minimum (and convert str to float)
    n = float(input("Minimum confidence rating (decimal between 0 and 1 inclusive): "))
    # filter mainDF for the minimum
    mainDf = mainDf[mainDf["confidence"] >= n]

# ultimately, drop confidence column
mainDf.drop(columns="confidence")
print(f"Final data set has {len(mainDf)} total entries.")

# features only!
pcaDf = mainDf.drop(columns="label")

# DATA ANALYSIS
# PCA
# find the top three principal components and return them as columns of mainDf
pca.PCA3(pcaDf, mainDf)

# T-SNE
# TODO: update t-sne to run on all features
# TODO: look into optimization
# Very computationally intensive! Likely will not run on average laptop!
# runs T-SNE on the top three principal components and returns the results as columns of mainDf
# keep the following line commented out to not run tsne
tsne.tsnePCA3(mainDf)

# DATA VISUALIZATION
# PCA
# visualizes the datatset using the top two principal components
pcaViz.viz(mainDf)

# T-TSNE
# visualizes the dataset using the t-sne results
# keep commented out if you haven"t uncommented the t-sne section
tsneViz.viz(mainDf)

# Pairplots
# TODO: Include functionality for an easier way to do an aggregate pairplot
# creates a pairplot for the specified lists of glitch(es) and features
glitches = ["Extremely_Loud", "Blip"]
features = ["centralFreq", "duration", "Q-value"]
pairplot.pPlot(mainDf, glitches, features)

print("Done!")
