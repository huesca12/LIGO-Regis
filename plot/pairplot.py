# IMPORTS
# import external packages
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,10))


def pPlot(mainDf, glitches, features):
    pDf = mainDf[mainDf.label.isin(glitches)]
    sns.pairplot(
        pDf,
        vars=features,
        hue='label',
        palette=sns.color_palette("hls", len(glitches))
        ).fig.savefig('pairplot.png')
