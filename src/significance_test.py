import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-r","--base")
    parser.add_argument("-s","--system")

    args = parser.parse_args()

    df_base = pd.read_csv(args.base, sep="\t", header=None, names=["@M","@5","@10"])
    df_system = pd.read_csv(args.system ,sep="\t", header=None, names=["@M","@5","@10"])

    print("@M: {}".format(ttest_ind(df_base["@M"], df_system["@M"])))
    print("@5: {}".format(ttest_ind(df_base["@5"], df_system["@5"])))
    print("@10: {}".format(ttest_ind(df_base["@10"], df_system["@10"])))