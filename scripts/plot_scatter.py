# Small script to generate a scatter plot and save it as PNG
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for saving files
import matplotlib.pyplot as plt
import numpy as np
import os

# Generate sample data (replace with your x, y if available)
np.random.seed(0)
x = np.random.normal(size=300)
y = np.random.normal(size=300)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(x, y, alpha=0.5)
ax.set_aspect('auto')  # equivalent to 'automatic' in intent
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Scatter plot (sample data)')

out_path = os.path.join(os.path.dirname(__file__), '..', 'scatter_figure.png')
# normalize path
out_path = os.path.abspath(out_path)
fig.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")
