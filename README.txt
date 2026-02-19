================================================================================
                    AI VOICE DEEPFAKE DETECTION ENGINE
================================================================================

Project Name: AI Voice Deepfake Detection Engine

================================================================================
PROBLEM STATEMENT
================================================================================

The rapid advancement of AI voice synthesis technology has made it increasingly 
difficult to distinguish between real and fake (deepfake) voice recordings. 
This project implements a machine learning solution to automatically detect 
whether an audio sample is a genuine human voice or an AI-generated deepfake.

================================================================================
DATASET USED
================================================================================

DEEP-VOICE DeepFake Voice Recognition Dataset
- Source: Kaggle
- Contains both REAL (original) and FAKE (AI-synthesized) voice samples
- Features: 40 MFCC (Mel-Frequency Cepstral Coefficients) extracted from audio

Dataset Statistics:
- Training samples: 51
- Test samples: 13
- Class distribution: Imbalanced (more fake than real samples)
- Solution: Class-weighted loss function applied during training

================================================================================
MODEL TYPE
================================================================================

PyTorch 1D CNN (Convolutional Neural Network)
- Input: 40 MFCC features per audio sample
- Architecture:
  * Conv1D (32 filters, kernel=3) + ReLU + MaxPool
  * Conv1D (64 filters, kernel=3) + ReLU + MaxPool
  * Flatten
  * Dense (64) + ReLU + Dropout (0.3)
  * Dense (1) + Sigmoid

Final Test Accuracy: ~92%

================================================================================
FOLDER STRUCTURE
================================================================================

AI-Voice-Deepfake-Engine/
├── app.py                      # Streamlit web UI
├── README.txt                  # This file
├── dataset/                    # Audio dataset
│   ├── real/                   # Real voice samples
│   └── fake/                   # Deepfake voice samples
├── models/                     # Trained models
│   ├── deepfake_model.pth      # PyTorch trained model
│   ├── X_train.npy             # Training features
│   ├── X_test.npy              # Test features
│   ├── y_train.npy             # Training labels
│   └── y_test.npy              # Test labels
├── src/                        # Source code
│   ├── model_definition.py     # CNN model architecture
│   ├── predict.py              # Terminal prediction script
│   ├── prepare_dataset.py      # Dataset preparation script
│   └── train_model.py          # Model training script
└── prediction_logs.txt         # Prediction logs (created after running predict.py)

================================================================================
HOW TO RUN - TERMINAL VERSION
================================================================================

1. Navigate to the project directory:
   cd AI-Voice-Deepfake-Engine

2. Run prediction on an audio file:
   python src/predict.py <path_to_wav_file>

   Examples:
   python src/predict.py dataset/real/biden-original.wav
   python src/predict.py dataset/fake/biden-to-obama.wav

3. Output format:
   ----------------------------------------
   Deepfake Probability: XX.X%
   Prediction: REAL or FAKE
   Risk Level: LOW/MEDIUM/HIGH
   Confidence: High/Moderate/Low
   ----------------------------------------

4. Results are logged to prediction_logs.txt

================================================================================
HOW TO RUN - WEB UI (Streamlit)
================================================================================

1. Install Streamlit (if not installed):
   pip install streamlit

2. Run the Streamlit app:
   streamlit run app.py

3. The web interface will open in your browser

4. Upload a .wav file to get instant deepfake detection results

================================================================================
TECHNICAL DETAILS
================================================================================

- Framework: PyTorch
- Features: 40 MFCC (Mel-Frequency Cepstral Coefficients)
- Training: 30 epochs with class-weighted BCE loss
- Model saved as: deepfake_model.pth (193 KB)
- No GPU required - runs on CPU

Risk Levels:
- 0-40%: LOW (Likely real voice)
- 40-70%: MEDIUM (Suspicious, requires further analysis)
- 70-100%: HIGH (Likely deepfake)

================================================================================
LICENSE & ACKNOWLEDGMENTS
================================================================================

This project is for educational and research purposes.
Dataset: DEEP-VOICE DeepFake Voice Recognition (Kaggle)

================================================================================
