# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:55:02 2021

@author: emapr
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import numpy as np

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

#%% Social welfare

sns.set_theme(style="whitegrid")
social_welfare = pd.read_csv('15bus_SW_V2.csv')

fig = plt.figure(figsize=(8, 4)) # uncomment to set space between boxes

ax = sns.boxplot(x="Liquidity", y="Social Welfare", hue="Test Case",
                 data=social_welfare, palette="Set3", linewidth=0.6, width=0.5)
# ax.legend(bbox_to_anchor=(1.01, 1),
#            borderaxespad=0)

h,l = ax.get_legend_handles_labels()
plt.legend(h[0:],l[0:],ncol=4, loc='upper center', 
           bbox_to_anchor=[0.5, 1.15], 
           columnspacing=0.7, labelspacing=0.0,
           handletextpad=0.3, handlelength=1.4,
           fancybox=False, shadow=False)

adjust_box_widths(fig, 0.9) # uncomment to set space between boxes

plt.savefig('15bus_SW_V2.pdf', format='pdf', bbox_inches='tight')

