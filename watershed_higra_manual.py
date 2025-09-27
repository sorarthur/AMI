# === 1. SETUP: Import Libraries and Load Image ===
import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk, remove_small_holes, remove_small_objects
from scipy import ndimage as ndi

# Load the image and convert it to grayscale for analysis.
try:
    image_path = 'imgs/tissue.JPG' # Make sure to use the correct path to your image
    color_image = io.imread(image_path)
except FileNotFoundError:
    print(f"Error: The image file was not found at '{image_path}'")
    exit()

image_size = color_image.shape[:2]
gray_image = color.rgb2gray(color_image)


# === 2. HIGRA SETUP: Create Graph and Edge Weights ===
# This graph and its weights, based on the original image, will be used
# by the final watershed algorithm inside the loop.
print("Creating graph and edge weights...")
graph = hg.get_4_adjacency_graph(image_size)
edge_weights = hg.weight_graph(graph, gray_image, hg.WeightFunction.L1)


# === 3. PRE-PROCESSING: Create a Clean Foreground Mask ===
# This mask will define the major components that we will process individually.
otsu_threshold = threshold_otsu(gray_image)
binary_mask = gray_image < otsu_threshold
selem = disk(2)
opened_mask = opening(binary_mask, selem)
labeled_mask = measure.label(opened_mask)
regions_props = measure.regionprops(labeled_mask)
min_area_threshold = 180.0
final_foreground_mask = np.copy(labeled_mask)
for prop in regions_props:
    if prop.area < min_area_threshold:
        final_foreground_mask[final_foreground_mask == prop.label] = 0
final_foreground_mask = final_foreground_mask > 0


# === 4. PRECISION SEGMENTATION: Divide and Conquer with Manual Markers ===

# 1. Label the main components so we can process them one by one.
main_components, num_components = measure.label(final_foreground_mask, return_num=True)

# 2. Prepare a blank canvas for the final image and a label offset.
final_segmented_image = np.zeros(image_size, dtype=np.int32)
label_offset = 0

print(f"Starting manual segmentation for {num_components} main components...")

# 3. Loop over each main component to process it individually.
for i in range(1, num_components + 1):
    current_mask = (main_components == i)
    
    # a. MANUAL MARKER SELECTION
    print(f"\n--- Manual Intervention Needed for Component {i} ---")
    print("An image window will appear. Click on the center of each sub-region.")
    print("When done, close the image window to continue.")

    component_visual = np.zeros_like(gray_image)
    component_visual[current_mask] = gray_image[current_mask]
    fig, ax = plt.subplots()
    ax.imshow(component_visual, cmap='gray')
    ax.set_title(f"Select markers for Component {i} and close the window")
    clicked_points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    
    # b. Convert the clicks into a labeled marker image.
    if len(clicked_points) > 0:
        peak_coords = np.array(clicked_points, dtype=int)
        local_peak_markers = np.zeros(image_size, dtype=bool)
        local_peak_markers[peak_coords[:, 1], peak_coords[:, 0]] = True
        local_labeled_markers, num_local_regions = ndi.label(local_peak_markers)
        print(f"You selected {num_local_regions} markers.")
    else:
        print("No markers selected. Treating as a single region.")
        local_labeled_markers = current_mask.astype(int)
        num_local_regions = 1

    # c. Apply the Higra seeded watershed using the full graph and original weights.
    partition_local = hg.watershed.labelisation_seeded_watershed(graph, edge_weights, local_labeled_markers)
    
    # Convert partition to a full-sized labeled image.
    if hasattr(partition_local, 'to_label_image'):
        segmentation_local_full_image = partition_local.to_label_image(image_size)
    else:
        segmentation_local_full_image = partition_local

    # d. Crucial Step: Constrain the result to the current component's mask.
    local_segmentation = np.zeros(image_size, dtype=np.int32)
    local_segmentation[current_mask] = segmentation_local_full_image[current_mask]
    
    # e. Add the result to our final image, ensuring unique labels.
    local_segmentation[local_segmentation > 0] += label_offset
    final_segmented_image += local_segmentation
    label_offset += num_local_regions

# === 5. POST-PROCESSING ===
# Remove the noises from the final segmented image.
print("\nPost-processing the final segmented image to remove small artifacts...")
min_final_area = 100
final_segmented_image_cleaned = np.zeros_like(final_segmented_image)
for label in range(1, label_offset + 1):
    component_mask = (final_segmented_image == label)
    component_area = np.sum(component_mask)
    if component_area >= min_final_area:
        final_segmented_image_cleaned[component_mask] = label
        final_segmented_image = final_segmented_image_cleaned



# === 6. AREA ANALYSIS ===
labels, areas = np.unique(final_segmented_image, return_counts=True)
total_tissue_area = 0
component_areas = {}

for label, area in zip(labels, areas):
    if label > 0:
        component_areas[label] = area
        total_tissue_area += area


# === 7. VISUALIZATION ===
print("\nProcess complete. Visualizing final result.")
# (The visualization code remains the same as the previous version)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Tissue Segmentation and Area Analysis (Manual Higra Method)', fontsize=20)
ax = axes.ravel()

ax[0].imshow(color_image)
ax[0].set_title("1. Original Image")
ax[0].axis('off')

ax[1].imshow(final_foreground_mask, cmap='gray')
ax[1].set_title("2. Initial Major Components")
ax[1].axis('off')

ax[2].imshow(final_segmented_image, cmap='nipy_spectral')
ax[2].set_title("3. Final Segmentation (Manual Markers)")
ax[2].axis('off')

ax[3].axis('off')
ax[3].set_title("4. Area Analysis")

if component_areas:
    sorted_items = sorted(component_areas.items())
    mid_point = len(sorted_items) // 2 + (len(sorted_items) % 2)
    left_column_items = sorted_items[:mid_point]
    right_column_items = sorted_items[mid_point:]
    left_column_str = "\n".join([f"Comp {num}: {area} px" for num, area in left_column_items])
    right_column_str = "\n".join([f"Comp {num}: {area} px" for num, area in right_column_items])
    ax[3].text(0.05, 0.80, left_column_str, fontsize=10, fontfamily='monospace', verticalalignment='top')
    ax[3].text(0.55, 0.80, right_column_str, fontsize=10, fontfamily='monospace', verticalalignment='top')
    total_area_str = f"\n\nTotal Tissue Area: {total_tissue_area} pixels"
    ax[3].text(0.30, 0.85, total_area_str, fontsize=10, fontweight='bold', verticalalignment='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()