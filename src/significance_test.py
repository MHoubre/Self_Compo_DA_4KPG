import pandas as pd
from scipy.stats import ttest_ind
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-r","--base")
    parser.add_argument("-s","--system")

    args = parser.parse_args()

    df_base = pd.read_csv(args.base, sep="\t", header=None, names=["all@M","all@5","all@10","pre@M","pre@5","pre@10","abs@M","abs@5","abs@10"])
    df_system = pd.read_csv(args.system ,sep="\t", header=None, names=["all@M","all@5","all@10","pre@M","pre@5","pre@10","abs@M","abs@5","abs@10"])

    print("ALL@M: {}".format(ttest_ind(df_base["all@M"], df_system["all@M"])))
    print("ALL@5: {}".format(ttest_ind(df_base["all@5"], df_system["all@5"])))
    print("ALL@10: {}".format(ttest_ind(df_base["all@10"], df_system["all@10"])))

    print("PRE@M: {}".format(ttest_ind(df_base["pre@M"], df_system["pre@M"])))
    print("PRE@5: {}".format(ttest_ind(df_base["pre@5"], df_system["pre@5"])))
    print("PRE@10: {}".format(ttest_ind(df_base["pre@10"], df_system["pre@10"])))

    print("ABS@M: {}".format(ttest_ind(df_base["abs@M"], df_system["abs@M"])))
    print("ABS@5: {}".format(ttest_ind(df_base["abs@5"], df_system["abs@5"])))
    print("ABS@10: {}".format(ttest_ind(df_base["abs@10"], df_system["abs@10"])))