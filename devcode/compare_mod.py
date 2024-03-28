import pandas as pd
import matplotlib.pyplot as plt

def comp():
    df_tree = pd.read_csv('data\output_dt.csv')
    df_log  = pd.read_csv('data\output_log.csv')

    plt.scatter(df_tree['score'],df_log['score'])
    plt.show()

    diff = df_tree['score'] - df_log['score']
    plt.hist(df_tree['score'] - df_log['score'],bins=100)
    plt.show()