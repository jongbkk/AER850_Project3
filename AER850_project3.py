import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the motherboard image
image_path = "C:\\Users\\user\\OneDrive\\Documents\\AER850\\motherboard_image.JPEG"
original_image = cv2.imread(image_path)

# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply GaussianBlur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 4: Use Canny edge detector for edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Step 5: Find contours in the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Filter out small contours based on area
min_contour_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Step 7: Create a mask and extract the PCB from the original image
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Step 8: Bitwise AND operation to extract the PCB
extracted_image = cv2.bitwise_and(original_image, original_image, mask=mask)

# Convert the extracted image to RGB
colored_extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB)

# Display the results using matplotlib (for Jupyter compatibility)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(colored_extracted_image)
plt.title("Extracted Image")

plt.show()
