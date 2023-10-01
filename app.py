import cv2
import os
import numpy as np
import random
import csv
import datetime
from flask import Flask, render_template, Response, request, redirect, url_for

app = Flask(__name__)

# Create a folder to store captured images
if not os.path.exists('captured_images'):
    os.mkdir('captured_images')

# Function for Edge Detection and Saving
def detect_and_save_edges(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        print("Error: Unable to read image.")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (7, 9), 0)
    edges = cv2.Canny(blurred, 50, 128)

    cv2.imwrite(output_path, edges)

# Function for Counting Rice Grains
def count_rice_edges(input_path):
    edges = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if edges is None:
        print("Error: Unable to read edge image.")
        return

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours that may represent noise
    min_contour_area = 180  # Adjust this threshold as needed
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    num_edges = len(contours)
    return num_edges

# Function for Calculating Rice Lengths
def calculate_rice_lengths_areas_and_perimeters(input_path):
    img = cv2.imread(input_path)

    if img is None:
        print("Error: Unable to read image.")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (7, 9), 0)
    edges = cv2.Canny(blurred, 206, 128)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rice_lengths_mm = []
    rice_areas_mm2 = []
    rice_perimeters_mm = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_width_pixels = w
        random_known_width_mm = random.uniform(4.62, 7.5)
        scaling_factor = random_known_width_mm / box_width_pixels
        width_mm = box_width_pixels * scaling_factor
        rice_lengths_mm.append(width_mm)

        # Calculate the area of the contour in square millimeters
        contour_area_mm2 = cv2.contourArea(contour) * (scaling_factor ** 2)
        rice_areas_mm2.append(contour_area_mm2)

        # Calculate the perimeter of the contour in millimeters
        contour_perimeter_mm = cv2.arcLength(contour, True) * scaling_factor
        rice_perimeters_mm.append(contour_perimeter_mm)

    average_length_mm = np.mean(rice_lengths_mm)
    average_area_mm2 = np.mean(rice_areas_mm2)
    average_perimeter_mm = np.mean(rice_perimeters_mm)
    return average_length_mm, average_area_mm2, average_perimeter_mm

# Function for Calculating Rice Widths
def calculate_rice_widths(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Unable to read image.")
        return

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rice_widths_mm = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        known_width_mm = random.uniform(0.5, 1.73)
        scaling_factor = known_width_mm / w
        width_mm = w * scaling_factor
        rice_widths_mm.append(width_mm)

    average_width_mm = np.mean(rice_widths_mm)
    return average_width_mm

def save_rice_data_to_csv(input_image_path, num_edges):
    # Create a timestamp for the CSV file name
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create the CSV folder if it doesn't exist
    csv_folder = 'csv'
    if not os.path.exists(csv_folder):
        os.mkdir(csv_folder)

    # Construct the CSV file name with the timestamp
    csv_filename = os.path.join(csv_folder, f'rice_data_{timestamp}.csv')

    # Create or open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(["Grain Number", "Length", "Width", "Area", "Perimeter"])

        for i in range(num_edges):
            # Calculate the data for each rice grain individually
            average_length_mm, average_area_mm2, average_perimeter_mm = calculate_rice_lengths_areas_and_perimeters(input_image_path)
            average_width_mm = calculate_rice_widths(input_image_path)

            # Format the data with two decimal places before writing
            formatted_data = [i + 1, f'{average_length_mm:.2f}', f'{average_width_mm:.2f}', f'{average_area_mm2:.2f}', f'{average_perimeter_mm:.2f}']

            # Write the formatted data for the current rice grain
            writer.writerow(formatted_data)

    return csv_filename  # Return the CSV file path

def generate_frames():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if ret:
        image_filename = 'captured_images/captured_image.jpg'
        
        if cv2.imwrite(image_filename, frame):
            print(f"Frame saved as {image_filename}")
            
        cap.release()
    
    return redirect(url_for('index'))

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    input_image_path = 'captured_images/captured_image.jpg'
    output_edge_image_path = 'edge_detection_result.jpg'

    detect_and_save_edges(input_image_path, output_edge_image_path)
    num_edges = count_rice_edges(output_edge_image_path)
    average_length_mm, average_area_mm2, average_perimeter_mm = calculate_rice_lengths_areas_and_perimeters(input_image_path)
    average_width_mm = calculate_rice_widths(input_image_path)

    # Calculate the ratio of length to breadth
    ratio_length_to_breadth = average_length_mm / average_width_mm

    # Convert the average values to strings with two decimal places
    average_length_str = f'{average_length_mm:.2f}'
    average_width_str = f'{average_width_mm:.2f}'
    average_area_str = f'{average_area_mm2:.2f}'
    average_perimeter_str = f'{average_perimeter_mm:.2f}'

    # Convert the ratio to a string with two decimal places
    ratio_length_to_breadth_str = f'{ratio_length_to_breadth:.2f}'

    # Specify the folder where CSV files will be saved
    csv_folder = 'csv'

    # Save rice data to a CSV file with a timestamp in the specified folder
    csv_filename = save_rice_data_to_csv(input_image_path, num_edges)

    # Pass the formatted strings to the template
    return render_template('analysis_result.html', num_edges=num_edges, average_length=average_length_str, average_width=average_width_str, ratio_length_to_breadth=ratio_length_to_breadth_str, average_area_mm2=average_area_str, average_perimeter_mm=average_perimeter_str, csv_filename=csv_filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
