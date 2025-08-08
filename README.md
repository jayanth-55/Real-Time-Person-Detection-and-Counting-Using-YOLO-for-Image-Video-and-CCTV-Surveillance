# Real-Time-Person-Detection-and-Counting-Using-YOLO-for-Image-Video-and-CCTV-Surveillance
YOLO-based deep learning project for detecting and counting human heads in crowd images, with accuracy evaluation using metrics like MAE, RMSE, Precision, Recall, and F1-score. Suitable as a minor project for academic purposes.
# Crowd Counting and Head Detection using YOLO

This project implements **YOLO-based object detection** to perform **crowd counting and head detection** from images.  
It utilizes deep learning to automatically detect and count the number of people (or heads) in an image, and calculates performance metrics such as accuracy, precision, recall, and F1 Score.

---

## 📌 Features
- **YOLO object detection** for detecting heads in images.
- **Automatic crowd counting** from detection results.
- **Evaluation metrics**:
  - Overall Accuracy
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Precision, Recall, F1-score
- **Custom dataset support** (replace with your own images and ground truth counts).
- **Supports batch processing** of multiple images.

---

## 📂 Project Structure

crowd_count_head_yolo/
│
├── **input/ #images,videos**
├── **output/ #output of main.py for video and images input**
├── accuracy.py # calculates accuracy while predicting predictedcount, error, and absolute error
├── main.py #code for detecting person count realtime,input(image or video or webcam)
├── performance_metrics.py #calculates RMSE, MAE, Recall, Precision, F1-score
├── requirements.txt # Python dependencies
├── LICENSE # Project license
├── ground_truth.csv #images and their actual count 
├── results_with_accuracy.csv #created when running accuracy.py 
└── README.md # Project documentation
---

## 📦 Requirements
- **Python 3.8+**
- **PyTorch**
- **Ultralytics YOLOv8**
- **Pandas**
- **OpenCV**
- **NumPy**

---

## 📜 License
- **This project is licensed under the MIT License – see the LICENSE file for details.**

---

## 👨‍💻 Authors
- **Jayanth Kumar Yaramanedi – Project Lead**
- **Thummagunta Vasantha Rani – Data Preparation**
- **Vaddi Venkata Dinesh – Model Training & Testing**

---

## Example Output:
![seq_000024_annotated](https://github.com/user-attachments/assets/2f13de3d-78a4-4ff7-8cab-343bed3a328b)

---

## 💡 Future Work
- **Improve detection accuracy on dense crowds**
- **Train a custom YOLO model with a larger dataset**
- **Deploy as a web or mobile application for live counting**

---
