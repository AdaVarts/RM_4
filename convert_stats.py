import time
import os
import numpy as np
import pandas as pd
from collections import Counter
import re
import config
import csv

# model name -> metric -> value
results_stats = {}

with open('results.csv', 'r') as f:
    df = tuple(csv.reader(f))
columns = df[0]
metrics = columns[1:]
df = df[1:]
model_names = (r[0] for r in df[1:])

for row in df:
    #print(row)
    model = row[0]
    if "fine_tuned" not in model: continue
    model = re.sub(r'./fine_tuned_model/[0-9]+\.split/', 'fine_tuned/', model)
    if model not in results_stats: results_stats[model] = {metric: [] for metric in metrics}
    for i,v in enumerate(row):
        if i==0: continue # first col is not a metric
        v = re.sub(r'[()\[\]]|np\.float64', '', v)
        v = float(v) if v != '' else float('nan')

        metric = columns[i]
        results_stats[model][metric].append(v)
        #print(f'{model} -> {metric} -> {v}')
avgs = {}
metrics = tuple(metric for metric in metrics if "std" not in metric and "mean" not in metric)
print(metrics)
for model_name in results_stats.keys():
    avgs[model_name] = {}
    for metric in metrics:
        #print(results_stats[model_name][metric])
        avgs[model_name][f'{metric}_mean'] = np.mean(tuple(results_stats[model_name][metric]))
        avgs[model_name][f'{metric}_std'] = np.std(results_stats[model_name][metric])
        

df = pd.DataFrame.from_dict(avgs, orient='index')

print(df)
df.to_csv('results_stats.csv', mode='w', header=True)