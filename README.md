# Object Detection & Time Series Prediction Using YOLOv8 and LSTM

This project utilizes YOLOv8 for real-time object detection and tracking, combined with an LSTM model for time-series prediction based on collected data.

## 🔍 Overview
Using the pretrained YOLOv8 model `yolov8n.pt`, the system detects and tracks objects. The SORT algorithm by Alex Bewley ([alex@bewley.ai](mailto:alex@bewley.ai)) is used for tracking. The collected data is processed and analyzed using an LSTM model with `TimeseriesGenerator` to make future predictions.

## ⚡ Features
- Real-time object detection using YOLOv8
- Object tracking with SORT algorithm
- Time-series prediction with LSTM
- Pre-trained dataset sourced from Kaggle

## 💡 Usage
```bash
python YoloDetector.py   # Run YOLOv8 for object detection
python sort.py           # Perform object tracking
python LSTMPrediction.ipynb     # Train and evaluate LSTM model
```

## 🛠 Technologies Used
- **YOLOv8** - Object Detection
- **SORT** - Object Tracking
- **TensorFlow/Keras** - LSTM-based Time Series Prediction
- **Python** - Backend Implementation

## 📌 Future Improvements
- Optimize LSTM model for better predictions
- Enhance real-time processing efficiency
- Deploy as a web-based application
