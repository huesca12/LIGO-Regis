# IMPORTS
# import external packages
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 10))


def viz(mainDf):
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="label",
        palette=sns.color_palette("hls", 22),
        data=mainDf,
        legend="full",
        alpha=0.3
    ).get_figure().savefig('tsne.png')
