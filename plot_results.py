#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022-04-14 Junxian <He>
#
# Distributed under terms of the MIT license.


import argparse
import os
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import make_interp_spline, pchip

parser = argparse.ArgumentParser()

parser.add_argument('--search_dir', type=str, default='./', help='root directory where the results folders are')
parser.add_argument('--results_pathname', type=str, default='webtext_',
                    help='expected to be of the form [NAME]_[PERCENT]_percent_[LAMBDA], so make this value "[NAME]_"')
parser.add_argument('--baseline_path', type=str, default='vanilla_gpt2_xl', help='path with the baseline results')
parser.add_argument('--total_num_shards', type=int, default=200, help='total number of shards the data was split into')

args = parser.parse_args()

def plot_results(args, out_path, plot=True, metric='eval_real_ppl', ylabel='perplexity'):
    search_dir       = args.search_dir
    results_pathname = args.results_pathname
    baseline_path    = args.baseline_path
    total_num_shards = args.total_num_shards

    immediate_subdirectories = [name for name in os.listdir(search_dir)
                                if os.path.isdir(os.path.join(search_dir, name))] # get immediate subdirectories
    immediate_subdirectories = [name for name in immediate_subdirectories
                                if results_pathname in name and 'percent' in name] # filter by ones with the results pathname in it

    # from the path names, figure out how many different lambda values and percents were used
    seen_percents, seen_lambdas = set(), set()
    for name in immediate_subdirectories:
        tokens = name.split('_')
        assert tokens[0] == results_pathname[:-1] and tokens[2].lower() == 'percent' # extra sanity check

        curr_percent, curr_lambda = int(tokens[1]), float(tokens[3])
        seen_percents.add(curr_percent)
        seen_lambdas.add(curr_lambda)

    seen_percents, seen_lambdas = list(seen_percents), list(seen_lambdas) # turn into lists to index into them later
    seen_percents, seen_lambdas = sorted(seen_percents), sorted(seen_lambdas) # sort them to ensure they're in order
    used_percents, used_lambdas = len(seen_percents), len(seen_lambdas) # number of unique percent and lambda values

    # collect real_ppl vs percentage of shards number used for many diff lambda values
    # each row is for a different lambda value, and each row has `used_percents` entries,
    # each containing the real_ppl for that amount of used percents
    results = np.zeros((used_lambdas, used_percents))
    for name in immediate_subdirectories:
        assert len(os.listdir(name)) == 2 # there should only be two files: all_results.json and eval_results.json

        # find the current percentage and current lambda value, and get their index in the previous list
        tokens = name.split('_')
        curr_percent, curr_lambda = int(tokens[1]), float(tokens[3])
        percent_idx, lambda_idx = seen_percents.index(curr_percent), seen_lambdas.index(curr_lambda)

        # store the real ppl
        json_path = os.path.join(name, 'eval_results.json')
        f = open(json_path, 'r')
        data = json.load(f)
        results[lambda_idx, percent_idx] = data[metric]
        f.close()

    # get baseline value
    f = open(os.path.join(baseline_path, 'eval_results.json'), 'r')
    data = json.load(f)
    baseline = data[metric]
    f.close()

    # HARDCODED -- get gpt2-large value
    f = open('vanilla_gpt2_large/eval_results.json', 'r')
    data = json.load(f)
    another_baseline = data[metric]
    f.close()

    if plot:
        # baseline values are plotted as horizontal lines
        plt.axhline(y=baseline, color='cyan', label='gpt2-xl', linestyle='--')
        plt.axhline(y=another_baseline, color='magenta', label='gpt2-large', linestyle='--')

        # interpolate & plot
        x = np.array(seen_percents)
        x_new = np.linspace(x.min(), x.max(), 300)
        for i in range(used_lambdas):
            y = results[i]
            spline = pchip(x, y) #k=len(y) - 1)
            smooth_y = spline(x_new)

            plt.plot(x_new, smooth_y, label=f'lmbda = {seen_lambdas[i]}')
            plt.scatter(x, y) #c=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])

        plt.xlabel(f'percentage of webtext used as datastore')
        plt.ylabel(ylabel)
        plt.legend(prop={'size': 6})
        #plt.title(f'{ylabel} vs percent of webtext used as datastore')

        plt.savefig(out_path)

    return seen_lambdas, results, baseline, another_baseline

seen_lambdas, results, baseline, another_baseline = plot_results(args, 'fig_webtext.png', plot=True)

# display
print(f'gpt2-xl real ppl = {baseline}')
print(f'gpt2-large real ppl = {another_baseline}')
for i in range(len(seen_lambdas)):
    print(f'lmbda = {seen_lambdas[i]}: {list(results[i])}')




