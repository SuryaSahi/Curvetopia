import numpy as np
import cv2
import os
import cairosvg
from PIL import Image
from io import BytesIO

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

def svg_to_png(svg_path, output_path):
    try:
        cairosvg.svg2png(url=svg_path, write_to=output_path)
        print(f"SVG converted to PNG and saved at {output_path}.")
    except Exception as e:
        print(f"An error occurred while converting SVG to PNG: {e}")

def classify_shape(contour):
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 2:
        return "Straight line"
    elif num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            return "Rounded rectangle"
        else:
            return "Rectangle"
    elif num_vertices == 5:
        return "Regular Pentagon"
    elif num_vertices > 6:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.8:
            return "Circle"
        elif num_vertices > 10:
            return "Star shape"
        else:
            return "Regular Polygon"

    return "Unknown Shape"

def detect_shapes(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Image is empty or could not be loaded.")
        
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
    # Paths to the CSV file and SVG image
    csv_path = 'C:\\Users\\ASUS\\Desktop\\Curvetopia\\problems\\frag0.csv'
    svg_path = 'C:\\Users\\ASUS\\Desktop\\Curvetopia\\problems\\frag0.svg'
    png_path = 'C:\\Users\\ASUS\\Desktop\\Curvetopia\\problems\\frag0.png'
    
    # Process the CSV file
    if os.path.exists(csv_path):
        print(f"Processing CSV file at {csv_path}.")
        read_csv(csv_path)
    else:
        print(f"The CSV file does not exist at {csv_path}.")
    
    # Convert SVG to PNG
    if os.path.exists(svg_path):
        print(f"Processing the SVG file at {svg_path}.")
        svg_to_png(svg_path, png_path)
        # Process the converted PNG file
        detect_shapes(png_path)
    else:
        print(f"The SVG file does not exist at {svg_path}.")

if __name__ == "__main__":
    main()
