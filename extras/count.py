import numpy
import pandas
import operator

mainDF = pandas.read_csv("data/gspy_o3a.csv")

h1DF = mainDF[mainDF["ifo"] == "H1"]
h1total = len(h1DF)

l1DF = mainDF[mainDF["ifo"] == "L1"]
l1total = len(l1DF)

glitches = mainDF["label"].unique()

h1_counts = []
h1percentages = []

l1_counts = []
l1percentages = []

for glitch in glitches:
    h1_glitch = len(h1DF[h1DF["label"] == glitch])
    l1_glitch = len(l1DF[l1DF["label"] == glitch])
    h1_counts.append(h1_glitch)
    l1_counts.append(l1_glitch)
    h1percentages.append(h1_glitch / h1total * 100)
    l1percentages.append(l1_glitch / l1total * 100)


h1 = sorted(list(zip(glitches, h1_counts, h1percentages)), key=operator.itemgetter(2), reverse=True)
l1 = sorted(list(zip(glitches, l1_counts, l1percentages)), key=operator.itemgetter(2), reverse=True)

h1DF = pandas.DataFrame(h1, columns=["H1 o3a", "Count", "%"])
h1DF = h1DF.append(h1DF.sum(numeric_only=True), ignore_index=True).replace(numpy.nan, "Total")
print(h1DF.to_string(index=False), end="\n\n\n")

l1DF = pandas.DataFrame(l1, columns=["L1 o3a", "Count", "%"])
l1DF = l1DF.append(l1DF.sum(numeric_only=True), ignore_index=True).replace(numpy.nan, "Total")
print(l1DF.to_string(index=False))

h1DF.to_csv("data/H1_o3a_counts.csv", index=False)
l1DF.to_csv("data/L1_o3a_counts.csv", index=False)
