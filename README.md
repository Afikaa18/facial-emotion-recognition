# Real-Time Facial Emotion Recognition for Behavioral Analysis

An AI-driven computer vision pipeline designed to analyze human behavior through video streams. By tracking face bounding boxes, extracting facial landmarks, and identifying dynamic shifts in micro-expressions, this system automatically classifies underlying human emotions in real-time. It translates raw visual inputs into measurable emotional data, making it highly effective for audience response tracking, behavioral research, and engagement metrics.


## Key Value Propositions
*   **Real-Time Emotion Detection:** Identifies multiple human faces and classifies their corresponding expressions frame by frame on a live timeline.
*   **Engagement Analysis:** Tracks micro-expression fluctuations over prolonged intervals to graph engagement metrics and overall emotional responses.
*   **Wide Application Scope:** Built to provide insights for diverse industry verticals, including modern education platforms, physical/virtual exhibitions, marketing analytics, and consumer research.
*   **Decision Support System:** Translates complex behavioral video analytics into concrete, readable datasets to support human-centered business strategy and academic research.


## Methodology & Workflow

The architecture of this project tracks your pipeline step-by-step from raw file ingest to output:
- Video Input (.MOV/.MP4)
- Frame Extraction via OpenCV 
- MTCNN Multi-Task Face Detection (Locates face bounding boxes)
- FER Feature Extraction Engine (Analyzes facial muscle landmarks)
- Dominant Emotion Classification (Computes highest probability emotion)
- OpenCV Video Stream Overlay (Draws visual green boxes & labels)
- Final Processed Video Export

## Core System Components

This architecture relies on robust deep learning frameworks and image processing dependencies:
*   **Python & OpenCV:** Handles underlying file system manipulation, real-time video stream manipulation, frame-by-frame processing, and font/box asset overlays.
*   **MTCNN (Multi-Task Cascaded Convolutional Networks):** Selected for high-accuracy face discovery, bounding box generation, and fast detection across dense crowds.
*   **FER (Facial Emotion Recognition):** The foundational model framework used to evaluate extracted face arrays against pre-trained facial behavior weights.
*   **TensorFlow:** Powers the neural network backends, enabling hardware acceleration via CUDA-enabled GPUs for heavy batch-frame calculations.
This project is written to run seamlessly inside a Google Colab notebook environment with GPU acceleration support enabled.

## Environment Prerequisites

To run this workflow on a local machine rather than Google Colab, make sure you have the following system libraries setup via your terminal environment:

```bash
pip install fer opencv-python matplotlib tensorflow
```


