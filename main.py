"""
Speech Emotion Recognition on RAVDESS-style dataset
- Classical ML: SVM, Random Forest (MFCC-based)
- Deep Learning: MLP (dense neural network on MFCCs)
- Kaggle-style visualisations: waveform, spectrogram, MFCC, chroma, confusion matrices.

Inspired by:
Shivam Burnwal, "Speech Emotion Recognition", Kaggle, 2020.
"""

# 0. REPRODUCIBILITY & ENVIRONMENT
import os
import random
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "0"

random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

# 1. IMPORTS
import glob
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# 2. CONFIGURATION

DATA_PATH = "audio_speech_actors_01-24"

SAMPLE_RATE = 22050
N_MFCC = 40
TEST_SIZE = 0.2
RANDOM_STATE = 42

plt.style.use("default")

# 3. IMAGE SAVE DIRECTORY
IMAGE_DIR = "Images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def save_fig(name):
    """
    Save current matplotlib figure to Images/<name>.png
    (high-res, tight layout)
    """
    path = os.path.join(IMAGE_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {path}")

# 4. LABEL PARSING (RAVDESS-STYLE FILENAMES)
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def get_emotion_from_filename(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    parts = file_name.split("-")
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, "unknown")

# 5. LOAD FILE PATHS
def load_file_paths(data_path: str = DATA_PATH):
    pattern = os.path.join(data_path, "Actor_*", "*.wav")
    file_list = glob.glob(pattern)
    print(f"Found {len(file_list)} audio files.")
    return file_list

# 6. FEATURE EXTRACTION (MFCC DATAFRAME)
def build_mfcc_dataframe(file_list, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    For each .wav:
      - load & trim
      - compute MFCCs
      - take mean across time
      - append to dataframe with 'emotion' label
    """
    rows = []
    for path in file_list:
        emotion = get_emotion_from_filename(path)
        try:
            signal, sr = librosa.load(path, sr=sample_rate)
            signal, _ = librosa.effects.trim(signal)

            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = mfcc.mean(axis=1)  # shape (n_mfcc,)

            row = mfcc_mean.tolist()
            row.append(emotion)
            rows.append(row)
        except Exception as e:
            print("Error processing", path, ":", e)

    cols = [f"mfcc_{i}" for i in range(n_mfcc)] + ["emotion"]
    df = pd.DataFrame(rows, columns=cols)
    print("MFCC DataFrame shape:", df.shape)
    print(df.head())
    return df

# 7. VISUALISATION HELPERS (KAGGLE-STYLE) – ALL SAVE TO Images/
def plot_example_waveform(file_path, sample_rate=SAMPLE_RATE):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(f"Waveform: {os.path.basename(file_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    save_fig("waveform")
    plt.show()

def plot_example_melspectrogram(file_path, sample_rate=SAMPLE_RATE):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    signal, _ = librosa.effects.trim(signal)

    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-Spectrogram")
    save_fig("mel_spectrogram")
    plt.show()

def plot_example_mfcc(file_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    signal, _ = librosa.effects.trim(signal)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title("MFCC Features")
    save_fig("mfcc")
    plt.show()

def plot_example_chroma(file_path, sample_rate=SAMPLE_RATE):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    signal, _ = librosa.effects.trim(signal)
    stft = np.abs(librosa.stft(signal))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", sr=sr)
    plt.colorbar()
    plt.title("Chroma Features")
    save_fig("chroma")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    save_fig(filename)
    plt.show()

# 8. CLASSICAL MODELS (SVM & RANDOM FOREST)
def run_classical_models(df):
    print("\n=== Classical Models: SVM & Random Forest (MFCC) ===")

    X = df.drop("emotion", axis=1).values
    y = df["emotion"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----- SVM -----
    print("\nTraining SVM (RBF kernel)...")
    svm_clf = SVC(kernel="rbf", C=10, gamma="scale")
    svm_clf.fit(X_train_scaled, y_train)
    y_pred_svm = svm_clf.predict(X_test_scaled)

    svm_acc = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

    plot_confusion_matrix(
        y_test,
        y_pred_svm,
        class_names=le.classes_,
        title="SVM Confusion Matrix",
        filename="cm_svm",
    )

    # ----- Random Forest -----
    print("\nTraining Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

    plot_confusion_matrix(
        y_test,
        y_pred_rf,
        class_names=le.classes_,
        title="Random Forest Confusion Matrix",
        filename="cm_rf",
    )

    return {
        "label_encoder": le,
        "scaler": scaler,
        "svm": svm_clf,
        "rf": rf_clf,
        "X_test": X_test,
        "y_test": y_test,
        "svm_acc": svm_acc,
        "rf_acc": rf_acc,
    }

# 9. MLP (DENSE NN) ON MFCCs
def build_mlp(input_dim, num_classes):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

def run_mlp_on_mfcc(df):
    print("\n=== MLP on MFCC Features ===")

    X = df.drop("emotion", axis=1).values
    y = df["emotion"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)
    num_classes = y_cat.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_mlp(X_train.shape[1], num_classes)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nMLP Test Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nMLP Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=le.classes_,
        title="MLP Confusion Matrix",
        filename="cm_mlp",
    )

    # Training curves
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLP Accuracy")
    plt.legend()
    save_fig("mlp_accuracy")
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Loss")
    plt.legend()
    save_fig("mlp_loss")
    plt.show()

    return {
        "model": model,
        "history": history,
        "label_encoder": le,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "test_acc": acc,
    }

# 10. MAIN
if __name__ == "__main__":
    # 1. Load all .wav paths
    files = load_file_paths(DATA_PATH)

    # 2. Plot Kaggle-style audio visuals for ONE example file
    example_file = files[0]
    print("\nExample file:", example_file)
    plot_example_waveform(example_file)
    plot_example_melspectrogram(example_file)
    plot_example_mfcc(example_file)
    plot_example_chroma(example_file)

    # 3. Build MFCC feature DataFrame
    df = build_mfcc_dataframe(files)

    # 4. Classical baselines (SVM, RF) + confusion matrices
    classical_results = run_classical_models(df)

    # 5. MLP on MFCCs + confusion matrix + training curves
    mlp_results = run_mlp_on_mfcc(df)

    print("\n=== SUMMARY ===")
    print(f"SVM Accuracy:  {classical_results['svm_acc']:.4f}")
    print(f"RF Accuracy:   {classical_results['rf_acc']:.4f}")
    print(f"MLP Accuracy:  {mlp_results['test_acc']:.4f}")
