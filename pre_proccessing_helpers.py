# ======= Packages ========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ======= Functions ========
def check_values_count(data):
    """Use pandas value_counts function for each column in the Dataframe


    Parameters
    ----------

    data: A pandas DataFrame


    Returns
    -------

    None


    """

    for column in data:
        if column == 'test_date':
            continue
        count = data[column].value_counts()
        print(f"the different values for {column}:\n {count} \n")



def plot_dataset(data, plot_object, object_name):
    """Plot the data according to given plot object


    Parameters
    ----------

    data: A pandas DataFrame 
    plot_object: An object for plotting
    object_name: A string for the object's name
    

    Return
    ------
    
    None


    """

    #data = data.copy(deep=True)
    fig, axs = plt.subplots(5, 4, figsize = (22.0, 16.0))
    i, j = (0, 0)
    for column in tqdm(data.columns):
        ax = axs[i,j]
        if column == 'AGE':
            sns.kdeplot(data=data, x='AGE', fill=True, ax=ax)

        else:
            ax = sns.countplot(x=column, data=data, ax=ax)
            #ax.bar_label(ax.containers[0], padding=1)

        if j == 3:
            i = i + 1
            j = 0
            continue
        j = j + 1

    fig.tight_layout()

    