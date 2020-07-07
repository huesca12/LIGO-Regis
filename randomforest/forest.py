# IMPORTS
# import external packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# DATA PREPARATION
# ask if the user wants to use gspy_o3a or gspy_o3b
query = input("Would you like to analyze the o3a or o3b data set? (a/b): ")
# make sure the user inputted either "a" or "b"
assert query == "a" or query == "b", "You must input either \"a\" or \"b\""
# define file path to particular dataset
filename = f"../data/gspy_o3{query}.csv"

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

# tags only!
y = mainDf["label"]

# features only!
X = mainDf.drop(columns="label")

# get our model
model = RandomForestClassifier()

# fit our model
model.fit(X, y)

# get our model score
print(model.score(X, y))
