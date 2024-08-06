# import numpy as np

# def read_csv(csv_path):
#     np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
#     path_XYs = []
#     for i in np.unique(np_path_XYs[:, 0]):
#         npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
#         XYs = []
#         for j in np.unique(npXYs[:, 0]):
#             XY = npXYs[npXYs[:, 0] == j][:, 1:]
#             XYs.append(XY)
#         path_XYs.append(XYs)
#     return path_XYs

# import numpy as np
# import os

# def read_csv(csv_path):
#     try:
#         np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
#         print("Loaded data:")
#         print(np_path_XYs)
        
#         path_XYs = []
#         for i in np.unique(np_path_XYs[:, 0]):
#             npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
#             print(f"Data for path {i}:")
#             print(npXYs)
            
#             XYs = []
#             for j in np.unique(npXYs[:, 0]):
#                 XY = npXYs[npXYs[:, 0] == j][:, 1:]
#                 print(f"Data for XY {j}:")
#                 print(XY)
#                 XYs.append(XY)
#             path_XYs.append(XYs)
        
#         return path_XYs
    
#     except Exception as e:
#         print(f"An error occurred: {e}")


# csv_path = 'C:\\Users\\ASUS\\Desktop\\Curvetopia\\problems\\frag0.csv'
# if os.path.exists(csv_path):
#     print(f"The file exists at {csv_path}.")
#     data = read_csv(csv_path)
# else:
#     print(f"The file does not exist at {csv_path}.")


import numpy as np
import cv2
import os

def read_csv(csv_path):
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
        print("Loaded data:")
        print(np_path_XYs)
        
        path_XYs = []
        for i in np.unique(np_path_XYs[:, 0]):
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            print(f"Data for path {i}:")
            print(npXYs)
            
            XYs = []
            for j in np.unique(npXYs[:, 0]):
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                print(f"Data for XY {j}:")
                print(XY)
                XYs.append(XY)
            path_XYs.append(XYs)
        
        return path_XYs
    
    except Exception as e:
        print(f"An error occurred: {e}")

def classify_shape(contour):
    # Approximate contour with a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Identify the shape based on the number of vertices
    if num_vertices == 2:
        return "Straight line"
    elif num_vertices == 3:
        return "Triangle"  # not specified, but useful for further shapes
    elif num_vertices == 4:
        # Check if it is a rectangle or a rounded rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            return "Rounded rectangle"
        else:
            return "Rectangle"
    elif num_vertices == 5:
        return "Regular Pentagon"  # not specified, but useful for further shapes
    elif num_vertices > 6:
        # Approximate a circle with many vertices
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.8:
            return "Circle"
        elif num_vertices > 10:
            return "Star shape"  # This is a simplification for star-like shapes
        else:
            return "Regular Polygon"

    return "Unknown Shape"

def detect_shapes(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            shape = classify_shape(contour)
            print(f"Detected shape: {shape}")

    except Exception as e:
        print(f"An error occurred while detecting shapes: {e}")

def main():
    # Path to the CSV file and image file
    csv_path = 'C:\\Users\\ASUS\\Desktop\\Curvetopia\\problems\\frag0.csv'
    image_path = 'C:\\Users\\ASUS\\Desktop\\Curvetopia\\problems\\frag0.svg'
    
    # Process the CSV file
    if os.path.exists(csv_path):
        print(f"Processing CSV file at {csv_path}.")
        read_csv(csv_path)
    else:
        print(f"The CSV file does not exist at {csv_path}.")
    
    # Process the image file
    if os.path.exists(image_path):
        print(f"Processing the image at {image_path}.")
        detect_shapes(image_path)
    else:
        print(f"The image does not exist at {image_path}.")

if __name__ == "__main__":
    main()
