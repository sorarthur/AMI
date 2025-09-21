# --- 1. SETUP: Import Libraries and Load Image ---
import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, dilation, disk
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes

# Load the image and convert it to grayscale for analysis.
try:
    image_path = 'imgs/tissue.JPG' # Make sure to use the correct path to your image
    color_image = io.imread(image_path)
except FileNotFoundError:
    print(f"Error: The image file was not found at '{image_path}'")
    exit()

gray_image = color.rgb2gray(color_image)


# --- 2. HIGRA SETUP: Create Graph and Edge Weights ---
# This graph and its weights, based on the original image, will be used
# by the final watershed algorithm.
image_size = gray_image.shape
graph = hg.get_4_adjacency_graph(image_size)
edge_weights = hg.weight_graph(graph, gray_image, hg.WeightFunction.L1)


# --- 3. PRE-PROCESSING: Create a Clean Binary Mask ---
# Get an initial separation of tissue from the background using Otsu's method.
otsu_threshold = threshold_otsu(gray_image)
binary_mask = gray_image < otsu_threshold

# Define a structuring element ("paintbrush") for morphological operations.
selem = disk(2)

# Clean up small, noisy particles from the background using Morphological Opening.
opened_mask = opening(binary_mask, selem)


# --- 4. FILTERING: Remove Small Unwanted Objects by Area ---
# Find and label each distinct object (connected component) in the cleaned mask.
labeled_mask = measure.label(opened_mask)
regions_props = measure.regionprops(labeled_mask)

# Set a minimum area threshold to distinguish main components from fragments.
min_area_threshold = 180.0

# Create a filtered mask by keeping only the objects larger than the threshold.
filtered_mask = np.copy(labeled_mask)
for prop in regions_props:
    if prop.area < min_area_threshold:
        filtered_mask[filtered_mask == prop.label] = 0

binary_filtered_mask = filtered_mask > 0


# --- 5. MARKER CREATION: The Key to a Perfect Watershed ---
# a) Sure Foreground:
# Use the smart hole-filling algorithm. This creates a perfect foreground marker
# without any artificial bridges between components.
sure_foreground = binary_fill_holes(binary_filtered_mask)

# b) Sure Background:
# To find the "sure background", we slightly expand the foreground.
# Anything outside this expanded area is definitely the background.
dilated_foreground = dilation(sure_foreground, selem)
sure_background = (dilated_foreground == 0)

# c) Combine Markers into a Single Map:
# The watershed algorithm will use this map as its guide.
# Background will be 1, and each foreground component will be 2, 3, etc.
foreground_labels = measure.label(sure_foreground)

final_markers = np.zeros(gray_image.shape, dtype=np.int32)
final_markers[sure_background] = 1
final_markers[foreground_labels > 0] = foreground_labels[foreground_labels > 0] + 1


# --- 6. FINAL SEGMENTATION: Run the Guided Watershed ---
# Apply the seeded watershed algorithm from Higra.
# It uses the original graph, the original topography (weights), and our perfect guide map (markers).
final_partition = hg.watershed.labelisation_seeded_watershed(graph, edge_weights, final_markers)

# The result is a partition object, which already contains the final labeled image.
# In recent Higra versions, this is a NumPy array. In older ones, it might need conversion.
# This code handles both cases.
if hasattr(final_partition, 'to_label_image'):
    final_segmentation = final_partition.to_label_image(image_size)
else:
    final_segmentation = final_partition

final_component_count = final_markers.max() -1 # -1 to not count the background
print(f"Process complete. Watershed found {final_component_count} distinct components.")
# --- 7. AREA ANALYSIS ---
# Analyze the areas of the segmented components.
labels, area = np.unique(final_segmentation, return_counts=True)

total_tissue_area = 0

component_areas = {}

for label, area in zip(labels, area):
    # Skip the background label (0) and the sure background (1)
    if label > 1:
        component_number = label - 1  # Adjust label to match original component numbering
        component_areas[component_number] = area
        total_tissue_area += area

# --- 8. VISUALIZATION ---

# Display the original image, the cleaned binary mask, and the final segmentation, and the area analysis.  
# Create a 2x2 grid for the plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Tissue Segmentation and Area Analysis', fontsize=20)
ax = axes.ravel()

# --- Plotting the Images ---
ax[0].imshow(color_image)
ax[0].set_title("1. Original Image")
ax[0].axis('off')

ax[1].imshow(binary_filtered_mask, cmap='gray') # Using final_mask after hole filling
ax[1].set_title("2. Cleaned & Filtered Mask")
ax[1].axis('off')

ax[2].imshow(final_segmentation, cmap='nipy_spectral')
ax[2].set_title("3. Final Segmentation")
ax[2].axis('off')

# --- Formatting the Analysis Text ---
ax[3].axis('off')
ax[3].set_title("4. Area Analysis")

# Sort items for consistent ordering
sorted_items = sorted(component_areas.items())
mid_point = len(sorted_items) // 2 + (len(sorted_items) % 2) # Split into two columns

# Create the string for each column
left_column_items = sorted_items[:mid_point]
right_column_items = sorted_items[mid_point:]

left_column_str = "\n".join([f"Comp {num}: {area} px" for num, area in left_column_items])
right_column_str = "\n".join([f"Comp {num}: {area} px" for num, area in right_column_items])

# Position the two columns of text side-by-side
ax[3].text(0.05, 0.80, left_column_str, fontsize=12, fontfamily='monospace', verticalalignment='top')
ax[3].text(0.55, 0.80, right_column_str, fontsize=12, fontfamily='monospace', verticalalignment='top')

# Add the total area at the bottom
total_area_str = f"\n\nTotal Tissue Area: {total_tissue_area} pixels"
ax[3].text(0.05, 0.1, total_area_str, fontsize=14, fontweight='bold', verticalalignment='bottom')

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
plt.show()