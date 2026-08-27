import matplotlib.pyplot as plt
import numpy as np

# Create some sample data for the subplots
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a 3x3 subplot grid
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# Iterate through each row of subplots and set different background colors
colors = ['lightblue', 'lightgreen', 'lightcoral']  # Example colors for each row
for i, row in enumerate(axs):
    for ax in row:
        ax.plot(x, y)
        row_index = i % len(colors)  # Ensure cycling through colors for each row
        ax.set_facecolor(colors[row_index])

# Get the figure boundaries
fig_bbox = fig.bbox

# Draw rectangles covering the entire area of each row, including axis labels
for i, row in enumerate(axs):
    # Get the bounding box of each row in figure coordinates
    row_bbox = row[0].get_tightbbox(fig.canvas.get_renderer())
    rect = plt.Rectangle((fig_bbox.x0, row_bbox.y0), fig_bbox.width, row_bbox.height, color=colors[i % len(colors)])
    fig.patches.append(rect)

plt.tight_layout()
plt.show()
