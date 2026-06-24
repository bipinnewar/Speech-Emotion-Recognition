# Speech Emotion Recognition Using MFCC Features and Machine Learning

## Overview

Speech Emotion Recognition (SER) is the task of identifying human emotions from speech signals. This project implements a machine learning-based SER system that classifies speech recordings into different emotional categories using Mel-Frequency Cepstral Coefficients (MFCCs) and multiple classification algorithms.

The system was trained and evaluated using the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset and compares the performance of Support Vector Machine (SVM), Random Forest (RF), and Multilayer Perceptron (MLP) models.

---

## Features

* Audio preprocessing and silence removal
* MFCC feature extraction from speech recordings
* Emotion classification across 8 emotional categories
* Support Vector Machine (SVM) implementation
* Random Forest (RF) implementation
* Multilayer Perceptron (MLP) implementation
* Model performance evaluation using:

  * Accuracy scores
  * Confusion matrices
  * Learning curves
* Comparative analysis of machine learning models

---

## Dataset

### RAVDESS Dataset

The project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** dataset.

**Dataset Statistics**

* 1,440 speech recordings
* 24 professional actors (12 male, 12 female)
* 8 emotion classes:

  * Neutral
  * Calm
  * Happy
  * Sad
  * Angry
  * Fearful
  * Disgust
  * Surprised

Dataset Link:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## System Workflow

1. Load speech recordings
2. Resample audio to 22.05 kHz
3. Remove silence segments
4. Extract MFCC features
5. Generate fixed-length feature vectors
6. Normalize features
7. Split dataset into training and testing sets
8. Train machine learning models
9. Evaluate model performance

---

## Technologies Used

* Python
* NumPy
* Pandas
* Librosa
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn

---

## Feature Extraction

### MFCC (Mel-Frequency Cepstral Coefficients)

The system extracts 40 MFCC coefficients from each audio recording.

Feature extraction pipeline:

* Framing and Windowing
* Fast Fourier Transform (FFT)
* Mel Filter Bank Processing
* Log Compression
* Discrete Cosine Transform (DCT)
* Temporal Mean Aggregation

These features capture the spectral characteristics of speech signals that are useful for emotion recognition.

---

## Machine Learning Models

### Support Vector Machine (SVM)

* RBF Kernel
* C = 10
* Non-linear classification

### Random Forest (RF)

* 300 Decision Trees
* Bootstrap Aggregation
* Random Feature Selection

### Multilayer Perceptron (MLP)

* Hidden Layers: 256, 128
* ReLU Activation
* Dropout: 0.3
* Adam Optimizer
* 50 Training Epochs

---

## Results

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Support Vector Machine (SVM) | 66.67%   |
| Random Forest (RF)           | 57.99%   |
| Multilayer Perceptron (MLP)  | 57.64%   |

### Key Findings

* SVM achieved the highest overall accuracy.
* High-arousal emotions such as Angry, Fearful, and Surprised were classified more accurately.
* Low-arousal emotions such as Neutral, Calm, and Sad showed greater overlap and misclassification.
* MFCC features provide a strong baseline for speech emotion recognition but have limitations in capturing temporal dynamics.


---

## Authors

* Bipin Shrestha
* Sher Mian
* Md Minhajul Islam Ishrak
* Muhammad Saeed

---

## License

This project is intended for academic and research purposes.
