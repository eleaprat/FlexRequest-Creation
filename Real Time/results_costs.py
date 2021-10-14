# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:57:52 2021

@author: emapr
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid")

# load dataset
dso_costs = pd.read_csv('15bus_costs_V1.csv')

ax = sns.catplot(x="When", y="DSO Costs",
                 hue="Test Case", col="Liquidity",
                 data=dso_costs, palette="Set3", kind="bar",
                 height=4, aspect=.7, legend=False)

(ax.set_axis_labels("", "DSO Costs"))

plt.legend(loc='upper center')
# plt.subplots_adjust(top=.925)

plt.legend(bbox_to_anchor=(1.01, 0.65),
            borderaxespad=0)

# h,l = ax.get_legend_handles_labels()
# plt.legend(h[0:],l[0:],ncol=4, loc='upper center', 
#            bbox_to_anchor=[0.5, 1.15], 
#            columnspacing=0.7, labelspacing=0.0,
#            handletextpad=0.3, handlelength=1.4,
#            fancybox=False, shadow=False)

plt.savefig('15bus_costs_V1.pdf', format='pdf', bbox_inches='tight')