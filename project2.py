import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Normalize the image to range [0, 1]
img = img.astype(np.float64) / 255.0

# Define thresholds
T_Low = 0.075
T_High = 0.175

# Gaussian filter coefficients
B = np.array([
    [2, 4, 5, 4, 2],
    [4, 9, 12, 9, 4],
    [5, 12, 15, 12, 5],
    [4, 9, 12, 9, 4],
    [2, 4, 5, 4, 2]
], dtype=np.float64)
B /= 159.0  # Normalize filter

# Apply Gaussian filter
A = cv2.filter2D(img, -1, B)

# Sobel filters for horizontal and vertical edges
KGx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
KGy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

Filtered_X = cv2.filter2D(A, -1, KGx)
Filtered_Y = cv2.filter2D(A, -1, KGy)

# Compute gradient direction in degrees
arah = np.arctan2(Filtered_Y, Filtered_X) * 180 / np.pi
arah[arah < 0] += 360  # Convert negative angles to positive

# **MAPPING OUTPUT (Gradient Directions with Temperature-like Colors)**
plt.figure(figsize=(6, 6))
plt.imshow(arah, cmap='turbo')  # 'turbo' colormap has blue to red gradient
plt.colorbar(label="Direction (Degrees)")
plt.title('Gradient Direction Mapping')
plt.axis('off')
plt.show()

# Compute gradient magnitude
magnitude = np.sqrt(Filtered_X**2 + Filtered_Y**2)

# **High Thresholding Output**
T_High_Value = T_High * np.max(magnitude)
high_thresh_img = (magnitude >= T_High_Value).astype(np.uint8) * 255

plt.figure(figsize=(6, 6))
plt.imshow(high_thresh_img, cmap='gray')
plt.title('High Thresholding Output')
plt.axis('off')
plt.show()

# **Low Thresholding Output**
T_Low_Value = T_Low * np.max(magnitude)
low_thresh_img = (magnitude >= T_Low_Value).astype(np.uint8) * 255

plt.figure(figsize=(6, 6))
plt.imshow(low_thresh_img, cmap='gray')
plt.title('Low Thresholding Output')
plt.axis('off')
plt.show()

# **Final Edge Detection (Canny)**
T_res = np.zeros_like(magnitude)
for i in range(1, magnitude.shape[0] - 1):
    for j in range(1, magnitude.shape[1] - 1):
        if magnitude[i, j] < T_Low_Value:
            T_res[i, j] = 0
        elif magnitude[i, j] > T_High_Value:
            T_res[i, j] = 1
        elif (magnitude[i + 1, j] > T_High_Value or magnitude[i - 1, j] > T_High_Value or 
              magnitude[i, j + 1] > T_High_Value or magnitude[i, j - 1] > T_High_Value or 
              magnitude[i - 1, j - 1] > T_High_Value or magnitude[i - 1, j + 1] > T_High_Value or 
              magnitude[i + 1, j + 1] > T_High_Value or magnitude[i + 1, j - 1] > T_High_Value):
            T_res[i, j] = 1

# Convert to 8-bit image
edge_final = (T_res * 255).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(edge_final, cmap='gray')
plt.title('Final Edge Detection Output')
plt.axis('off')
plt.show()
