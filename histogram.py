#import external packages
import pandas as pd
import matplotlib.pyplot as plt

#Specify the label for analysis via user input
glitch_type = str(input("Enter the glitch label (e.g. Scattered_Light): "))
glitch = mainDf['label']==glitch_type
histDf = mainDf[glitch]

#Creates a histogram for a single feature
feature = str(input("Enter the feature to graph:"))
mainDf[feature].plot(kind='hist')

#Graph labeling and appearance 
plot_title = glitch_type
plt.title(plot_title)
plt.xlabel(feature)
plt.ylabel('')

plt.show()
