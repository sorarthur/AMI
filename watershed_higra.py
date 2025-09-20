import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, dilation, disk
from skimage.measure import regionprops

# ------------------ 1. SETUP: Load Image and Prepare Graph --------------------------------
# Load the image and convert it to grayscale for analysis.
image_path = 'imgs/tissue.JPG'
color_image = io.imread(image_path)
gray_image = color.rgb2gray(color_image)

# Create a 4-adjacency graph, which represents the pixel grid and their neighbors.
image_size = gray_image.shape
graph = hg.get_4_adjacency_graph(image_size)

# Calculate edge weights based on the intensity difference between neighboring pixels.
# This represents the "topography" for the watershed.
edge_weights = hg.weight_graph(graph, gray_image, hg.WeightFunction.L1)


# ------------------ 2. PRE-PROCESSING: Create a Clean Binary Mask --------------------------
# Get an initial separation of tissue (foreground) from the background.
otsu_threshold = threshold_otsu(gray_image)
binary_mask = gray_image < otsu_threshold

# Define a structuring element ("paintbrush") for morphological operations.
selem = disk(1)

# Clean up small, noisy particles from the background using Morphological Opening.
opened_mask = opening(binary_mask, selem)

# Fill small holes within the tissue areas using Morphological Closing.
closed_mask = closing(opened_mask, selem)


# ------------------ 3. OBJECT FILTERING: Keep Only Main Tissue Components ------------------
# Find and label each distinct object (connected component) in the cleaned mask.
labeled_mask = measure.label(closed_mask)
regions_props = regionprops(labeled_mask)

# Set a minimum area threshold to distinguish main components from small fragments.
# This value was determined by inspecting the areas of all found components.
min_area_threshold = 189.0

# Create the final foreground mask by keeping only the large objects.
final_foreground_mask = np.copy(labeled_mask)
for prop in regions_props:
    if prop.area < min_area_threshold:
        # "Erase" small objects by setting their pixels to 0.
        final_foreground_mask[final_foreground_mask == prop.label] = 0


# ------------------ 4. MARKER CREATION: Define Foreground and Background Markers --------------
# We will use the cleaned foreground mask to define markers for the watershed.
# The filtered mask gives us our "sure foreground" markers, already labeled.
foreground_markers = final_foreground_mask

# To find the "sure background", we slightly expand the foreground.
# Anything outside this expanded area is definitely the background.
dilated_foreground = dilation(final_foreground_mask, selem)
background_markers = (dilated_foreground == 0)

# Combine foreground and background markers into a single integer map.
# The watershed algorithm will use this map as its guide.
final_markers = np.zeros(gray_image.shape, dtype=np.int32)
# 1. Paint the sure background with the label '1'.
final_markers[background_markers] = 1
# 2. Paint the sure foreground with labels '2', '3', etc.
# (We add 1 to the existing labels to avoid conflict with the background label '1').
final_markers[foreground_markers > 0] = foreground_markers[foreground_markers > 0] + 1


# ------------------ 5. WATERSHED SEGMENTATION: Apply Seeded Watershed ------------------------
# Apply the seeded watershed algorithm from Higra.
# It uses the graph, the topography (weights), and our guide map (markers).
final_segmentation = hg.watershed.labelisation_seeded_watershed(graph, edge_weights, final_markers)

# ------------------ 6. VISUALIZATION: Display Results ---------------------------------------
# Display the original image and the final segmentation result.
plt.figure(figsize=(8, 8))
plt.imshow(final_segmentation, cmap='nipy_spectral')
plt.title('Final Guided Watershed Segmentation (Higra)')
plt.axis('off')
plt.show()