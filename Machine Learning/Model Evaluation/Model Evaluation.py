import pandas as pd 
from scikit_posthocs import posthoc_nemenyi_friedman
from statsmodels.sandbox.stats.multicomp import multipletests

data = pd.read_csv("algo_performance.csv") 
p = posthoc_nemenyi_friedman(data.values )

for i in range (0,5):
    print ("For the",i+1 ,"algorithm and a=0.05 with Bonferroni method the result is:",multipletests(p.values[:,i], method='bonferroni' ,alpha=0.05)[0])
    print ("For the",i+1 ,"algorithm and a=0.1 with Bonferroni method the result is:",multipletests(p.values[:,i], method='bonferroni' ,alpha=0.1)[0])
    print ("For the",i+1 ,"algorithm and a=0.25 with Bonferroni method the result is:",multipletests(p.values[:,i], method='bonferroni' ,alpha=0.25)[0])
    print()