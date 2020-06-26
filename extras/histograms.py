import matplotlib.pyplot as plt
import os
import pandas as pd

datasets = ["o3a", "o3b"]
sites = ["H1", "L1"]
glitch_types = ["Extremely_Loud", "Scratchy", "Power_Line", "Scattered_Light",
                "Koi_Fish", "Whistle", "Low_Frequency_Burst", "Blip",
                "Repeating_Blips", "Wandering_Line", "Violin_Mode",
                "None_of_the_Above", "Air_Compressor", "Tomte", "No_Glitch",
                "Low_Frequency_Lines", "Chirp", "Helix", "1400Ripples",
                "Light_Modulation", "Paired_Doves", "1080Lines"]
features = ["peakFreq", "snr", "amplitude", "centralFreq", "duration", "bandwidth", "Q-value"]

counter = 1
root_dir = "../histograms"
for dataset in datasets:

    data_file = f"../data/gspy_{dataset}.csv"
    mainDf = pd.read_csv(data_file)
    mainDf = mainDf[mainDf["confidence"] >= .9]
    dataset_dir = f"{root_dir}/{dataset}"
    os.mkdir(dataset_dir)

    for site in sites:

        siteDf = mainDf[mainDf["ifo"] == site]
        site_dir = f"{dataset_dir}/{site}"
        os.mkdir(site_dir)

        for glitch_type in glitch_types:

            glitch = siteDf["label"] == glitch_type
            histDf = siteDf[glitch]
            glitch_type_dir = f"{site_dir}/{glitch_type}"
            os.mkdir(glitch_type_dir)

            for feature in features:

                histDf[feature].plot(kind="hist", bins=100, figsize=(10, 10))
                title = glitch_type + " / " + feature
                plt.title(title)
                plt.xlabel(feature)
                plt.ylabel("Count")
                histogram_file = f"{glitch_type_dir}/{dataset}_{site}_{glitch_type}_{feature}"
                plt.savefig(histogram_file)
                plt.close()

                print(f"{counter}. Histogram generated! | {dataset} : {site} : {glitch_type} : {feature}")
                counter += 1
