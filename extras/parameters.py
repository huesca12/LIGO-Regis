import matplotlib.pyplot as plt
from math import inf
import pandas as pd

####################
DATA = "both"
# "o3a", "o3b", "both"
SITE = "both"
# "H1", "L1", "both"
GLITCH = ""
# "Extremely_Loud", "Scratchy", "Power_Line", "Scattered_Light",
# "Koi_Fish", "Whistle", "Low_Frequency_Burst", "Blip",
# "Repeating_Blips", "Wandering_Line", "Violin_Mode",
# "None_of_the_Above", "Air_Compressor", "Tomte", "No_Glitch",
# "Low_Frequency_Lines", "Chirp", "Helix", "1400Ripples",
# "Light_Modulation", "Paired_Doves", "1080Lines"
PLOT = True
# True, False

# If you do not want a minimum, use 0
# If you do not want a maximum, use inf
MIN_peakFreq = 0
MAX_peakFreq = inf
MIN_snr = 0
MAX_snr = inf
MIN_amplitude = 0
MAX_amplitude = inf
MIN_centralFreq = 0
MAX_centralFreq = inf
MIN_duration = 0
MAX_duration = inf
MIN_bandwidth = 0
MAX_bandwidth = inf
MIN_Qvalue = 0
MAX_Qvalue = inf


####################


def main():
    parameters = {
        "peakFreq": (MIN_peakFreq, MAX_peakFreq),
        "snr": (MIN_snr, MAX_snr),
        "amplitude": (MIN_amplitude, MAX_amplitude),
        "centralFreq": (MIN_centralFreq, MAX_centralFreq),
        "duration": (MIN_duration, MAX_duration),
        "bandwidth": (MIN_bandwidth, MAX_bandwidth),
        "Q-value": (MIN_Qvalue, MAX_Qvalue),
    }

    main_df = pd.concat([pd.read_csv(f"data/gspy_o3a.csv"), pd.read_csv(f"data/gspy_o3b.csv")], ignore_index=True) \
        if DATA == "both" else \
        pd.read_csv(f"data/gspy_{DATA}.csv")
    main_df = main_df if SITE == "both" else main_df[main_df["ifo"] == SITE]

    param_df = main_df
    for key, val in parameters.items():
        # key -> parameter | val[0] -> min | val[1] -> max
        param_df = param_df[(param_df[key] > val[0]) & (param_df[key] < val[1])]

    share = len(param_df[param_df["label"] == GLITCH]) / len(param_df) * 100
    recovery = len(param_df[param_df["label"] == GLITCH]) / len(main_df[main_df["label"] == GLITCH]) * 100
    number_distribution_string = "\t" + \
                                 param_df["label"].value_counts().to_string().replace("\n", "\n\t")
    percentage_distribution_string = "\t" + \
                                     param_df["label"].value_counts(normalize=True).to_string().replace("\n", "\n\t")

    print(f"Share: {share} %")
    print(f"Recovery: {recovery} %")
    print(
          "Full Breakdown:\n\n"
          "Number Distribution:\n"
          f"{number_distribution_string}\n\n"
          "Percentage Distribution:\n"
          f"{percentage_distribution_string}\n"
          )

    if PLOT:
        param_df["label"].value_counts(normalize=True).plot(kind="bar")
        plt.tight_layout()
        plt.show()


main()
