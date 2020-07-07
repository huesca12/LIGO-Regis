# IMPORTS
# import external packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import tree

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

# honest training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("Training random forest model!")

# get our model
model = RandomForestClassifier()

# fit our model
model.fit(X_train, y_train)

# get our out-of-bag estimator model score
print("The OOB-estimated traning set score is: {:f}".format(model.score(X, y)))

print("Training single decision tree!")

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

# visualize tree
fn=list(X.columns.values)
cn=list(y.unique())
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn,
               class_names=cn,
               filled = True);
fig.savefig('tree.png')
