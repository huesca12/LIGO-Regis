# IMPORTS
# import external packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# import local modules
import viz

# DATA PREPARATION
filename = f"data/gspy_o3a.csv"

# read the csv file into a dataframe
rawDf = pd.read_csv(filename)
oppDf = pd.read_csv(f"data/gspy_o3b.csv")

# select H1 or L1
rawDf = rawDf[rawDf["ifo"] == "H1"]
oppDf = oppDf[oppDf["ifo"] == "H1"]

# define the list of columns that we want to drop
# note the inclusion of peakFreq and amplitude (original TSNE plot we have did not account for these)
dropList = ["chisq", "chisqDof", "GPStime", "ifo", "imgUrl", "id"]
# drop the columns we don"t need
mainDf = rawDf.drop(columns=dropList)
oppDf = oppDf.drop(columns=dropList)

# ask if the user wants a minimum confidence rating
query = input("Would you like to drop data entries below a particular GSpy confidence rating? (y/n): ")
if query == "y":
    # if so, ask for the desired minimum (and convert str to float)
    n = float(input("Minimum confidence rating (decimal between 0 and 1 inclusive): "))
    # filter mainDF for the minimum
    mainDf = mainDf[mainDf["confidence"] >= n]
    oppDf = oppDf[oppDf["confidence"] >= n]

# ultimately, drop confidence column
mainDf.drop(columns="confidence")
print(f"Final data set has {len(mainDf)} total entries.")
oppDf.drop(columns="confidence")
print(f"Opposite data set has {len(oppDf)} total entries.")

# tags only!
y = mainDf["label"]
y_opp = oppDf["label"]

# features only!
X = mainDf.drop(columns="label")
X_opp = oppDf.drop(columns="label")

# honest training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state=0)

print("Training random forest model!")

# get our model
model = RandomForestClassifier()

# fit our model
model.fit(X_train, y_train)

# get our out-of-bag estimator model score
print("The OOB-estimated traning set score is: {:f}".format(model.score(X_test, y_test)))

print("Training single decision tree!")

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

print("The OOB-estimated traning set score is: {:f}".format(clf.score(X_test, y_test)))

# visualize tree
# ask if the user wants to run accuracy check on test set
query = input("Would you like to visualize the decision tree? (y/n): ")
if query == "y":
    viz.treeViz(X, y, clf)


# ask if the user wants to run accuracy check on test set
query = input("Would you like to check both models' accuracies on the other set? (y/n): ")
if query == "y":
    print("The OOB-estimated opposite set score for the random forest is: {:f}".format(model.score(X_opp, y_opp)))
    print("The OOB-estimated opposite set score for the decision tree is: {:f}".format(clf.score(X_opp, y_opp)))

# add new rows to dataframe
mainDf["Prediction"] = (model.predict(X))

def accuracy(row):
    if row['label'] == row['Prediction']:
        val = "Correct"
    else:
        val = "Incorrect"
    return val

def statistic(row):
    if (row['label'] == "Scattered_Light") & (row['Prediction'] == "Scattered_Light"):
        val = "True Positive"
    elif (row['label'] != "Scattered_Light") & (row['Prediction'] == "Scattered_Light"):
        val = "False Positive"
    elif (row['label'] != "Scattered_Light") & (row['Prediction'] != "Scattered_Light"):
        val = "True Negative"
    elif (row['label'] == "Scattered_Light") & (row['Prediction'] != "Scattered_Light"):
        val = "False Negative"
    else:
        val = "Error"
    return val

mainDf["Accuracy"] = mainDf.apply(accuracy, axis=1)
mainDf["Statistic"] = mainDf.apply(statistic, axis=1)

# calculate percentages
accuracy = (len(mainDf[mainDf['Accuracy'] == 'Correct']))/(len(mainDf))
tpr = (len(mainDf[mainDf['Statistic'] == 'True Positive']))/((len(mainDf[mainDf['Statistic'] == 'True Positive'])) + (len(mainDf[mainDf['Statistic'] == 'False Negative'])))
tnr = (len(mainDf[mainDf['Statistic'] == 'True Negative']))/((len(mainDf[mainDf['Statistic'] == 'True Negative'])) + (len(mainDf[mainDf['Statistic'] == 'False Positive'])))
far = (len(mainDf[mainDf['Statistic'] == 'False Positive']))/((len(mainDf[mainDf['Statistic'] == 'False Positive'])) + (len(mainDf[mainDf['Statistic'] == 'True Negative'])))
share = (len(mainDf[mainDf['Statistic'] == 'True Positive']))/((len(mainDf[mainDf['Statistic'] == 'True Positive'])) + (len(mainDf[mainDf['Statistic'] == 'False Positive'])))

print("Accuracy: " + str(accuracy))
print("TPR/Recovery: " + str(tpr))
print("TNR: " + str(tnr))
print("FAR: " + str(far))
print("Share: " + str(share))

# create csv file
mainDf.to_csv('resultsa.csv')

