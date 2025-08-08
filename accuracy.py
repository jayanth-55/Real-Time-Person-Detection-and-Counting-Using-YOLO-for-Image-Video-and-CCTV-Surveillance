from ultralytics import YOLO
import cv2
import pandas as pd
import os

# Load YOLO model
model = YOLO("yolov8n.pt")  # use 'yolov8m.pt' or 'yolov8l.pt' for better accuracy

# Paths
input_folder = "input"        # folder where your images are stored
csv_file = "ground_truth.csv" # your CSV file

# Load ground truth data
df = pd.read_csv(csv_file)

# To store results
pred_counts = []
true_counts = df['true_count'].tolist()
filenames = df['input'].tolist()

# Loop through each image in CSV
for filename in filenames:
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    # Run YOLO detection (only class 0 = person)
    results = model.predict(source=img, classes=[0], save=False, verbose=False)
    detected_count = len(results[0].boxes)

    pred_counts.append(detected_count)

# Add predictions to DataFrame
df['predicted_count'] = pred_counts

# Calculate error metrics
df['error'] = df['true_count'] - df['predicted_count']
df['absolute_error'] = abs(df['error'])
accuracy = 100 - (df['absolute_error'].sum() / df['true_count'].sum() * 100)

print(df)
print(f"\nOverall Accuracy: {accuracy:.2f}%")
