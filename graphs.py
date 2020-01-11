import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import datetime
import time


def get_averages_per_month(excell):
    # Index 1-12 for months
    # first list sum of the values in a specific month
    # second list represents the number of entries in that month
    columns_dictionary = {
        "Consumption_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Coal_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Gas_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Hidroelectric_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Nuclear_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Wind_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Solar_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Biomass_MW": [[0 for i in range(13)], [0 for i in range(13)]],
        "Production_MW": [[0 for i in range(13)], [0 for i in range(13)]]
    }

    for index in range(len(excell['Consumption_MW'])):
        month = pd.Timestamp(excell.at[index, 'Date'], unit='s').month
        for column in excell.columns[1:]:
            value = excell.at[index, column]
            columns_dictionary[column][0][month] += value
            columns_dictionary[column][1][month] += 1

    for key in columns_dictionary.keys():
        for index in range(1,13):
            try:
                columns_dictionary[key][0][index] /= columns_dictionary[key][1][index]
            except:
                continue
    return columns_dictionary

start_time = time.time() # just to check how long it takes to run :)

# Read Data
current_directory = os.path.dirname(sys.modules['__main__'].__file__)
file_location = os.path.join(current_directory, "data", "train_electricity.csv")
excell = pd.read_csv(file_location)[0:25000]

# AVERAGES PER MONTH FOR EACH CONSUMPTION TYPE
#----------------------------------------------
averages_dictionary = get_averages_per_month(excell)
print("TIME = ")
print(time.time() - start_time)

for key in averages_dictionary.keys():
    plt.plot(averages_dictionary[key][0],label = key, marker='o')
plt.legend()
plt.show()


# HEATMAP - Shows positive/negative correlation
# ----------------------------------------------
corr = excell.loc[:,excell.dtypes == 'float64'].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            cmap=sns.diverging_palette(250  , 15, as_cmap=True))
plt.show()
