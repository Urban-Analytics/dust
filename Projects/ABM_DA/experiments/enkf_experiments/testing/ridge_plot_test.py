# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(rc={'axes.facecolor': (0, 0, 0, 0)})

# Create data
rs = np.random.RandomState(666)
x = rs.randn(500)
g = np.random.choice([2, 5, 10, 20, 50, 100], size=500)
df = pd.DataFrame(dict(error=x, ensemble_size=g))

# Initialise facegrid
g = sns.FacetGrid(df, row='ensemble_size', hue='ensemble_size',
                  aspect=15, height=0.5)

# Draw densities
g.map(sns.kdeplot, 'error',
      bw_adjust=0.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, 'error', clip_on=False, color='w', lw=2, bw_adjust=0.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Function to label plots
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, 0.2, label, color=color,
            ha='left', va='center', transform=ax.transAxes)


g.map(label, 'error')

# Set plots to overlap
g.fig.subplots_adjust(hspace=0.25)

# Remove axes details that don't work with overlap
g.set_titles('')
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.savefig('ridge_test.pdf')
