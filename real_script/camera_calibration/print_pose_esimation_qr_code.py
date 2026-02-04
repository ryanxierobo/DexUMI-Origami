import cv2
import numpy as np

# Create the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Set the marker ID and size
marker_id = 23  # You can change this to any ID between 0 and 49
marker_size = 1000  # Size of the marker image in pixels

# Generate the marker
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Add a white border around the marker for better detection
border_size = 50
bordered_marker = cv2.copyMakeBorder(
    marker_image, 
    border_size, border_size, border_size, border_size, 
    cv2.BORDER_CONSTANT, 
    value=[255, 255, 255]
)

# Save the marker image
filename = f"aruco_marker_{marker_id}_printable.png"
cv2.imwrite(filename, bordered_marker)

print(f"ArUco marker with ID {marker_id} has been saved as {filename}")
print("You can now print this image for use with pose estimation.")

# Optionally display the marker (you can comment this out if not needed)
cv2.imshow("Printable ArUco Marker", bordered_marker)
cv2.waitKey(0)
cv2.destroyAllWindows()