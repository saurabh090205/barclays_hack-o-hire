"""
AI Voice Deepfake Detection - Prediction Script
This script loads a trained model and predicts whether an audio file is real or fake.
"""

import os
import sys
import wave
import numpy as np
import torch

# Import model from model_definition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_definition import DeepfakeCNN

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'deepfake_model.pth')
LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'prediction_logs.txt')
MFCC_FEATURES = 40


def load_wav_file(file_path):
    """Load a WAV file."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            framerate = wav_file.getframerate()
            
            raw_data = wav_file.readframes(n_frames)
            
            if sample_width == 1:
                data = np.frombuffer(raw_data, dtype=np.uint8)
                data = (data - 128) / 128.0
            elif sample_width == 2:
                data = np.frombuffer(raw_data, dtype=np.int16)
                data = data.astype(np.float32) / 32768.0
            elif sample_width == 4:
                data = np.frombuffer(raw_data, dtype=np.int32)
                data = data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            if n_channels == 2:
                data = data.reshape(-1, 2).mean(axis=1)
            
            return data, framerate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def mel_filterbank(n_mels, n_fft, sample_rate):
    """Create mel filterbank."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    low_freq = 0
    high_freq = sample_rate / 2
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left = bins[i]
        center = bins[i + 1]
        right = bins[i + 2]
        for j in range(left, center):
            fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i, j] = (right - j) / (right - center)
    
    return fbank


def scipy_dct(x, type=2, axis=0, norm=None):
    """Simplified DCT implementation (Type II)."""
    x = np.asarray(x)
    if axis != 0:
        x = np.swapaxes(x, 0, axis)
    
    N = x.shape[0]
    result = np.zeros_like(x)
    
    for k in range(N):
        for n in range(N):
            result[k] += x[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    
    if norm == 'ortho':
        result[0] *= np.sqrt(1 / (4 * N))
        result[1:] *= np.sqrt(1 / (2 * N))
    
    if axis != 0:
        result = np.swapaxes(result, 0, axis)
    
    return result


def compute_mfcc(y, sample_rate, n_mfcc=MFCC_FEATURES):
    """Compute MFCC features from audio signal."""
    try:
        n_fft = 2048
        hop_length = 512
        n_mels = 40
        
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), mode='constant')
        
        n_frames = 1 + (len(y) - n_fft) // hop_length
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start + n_fft]
            window = np.hanning(n_fft)
            frame = frame * window
            stft[:, i] = np.fft.rfft(frame)
        
        power_spec = np.abs(stft) ** 2
        
        fbank = mel_filterbank(n_mels, n_fft, sample_rate)
        mel_spec = np.dot(fbank, power_spec)
        
        log_mel_spec = np.log(mel_spec + 1e-10)
        
        mfccs = scipy_dct(log_mel_spec, type=2, axis=0, norm='ortho')[:n_mfcc]
        
        mfccs_mean = np.mean(mfccs, axis=1)
        
        return mfccs_mean
    except Exception as e:
        print(f"Error computing MFCC: {e}")
        return None


def extract_mfcc_features(y, sr, n_mfcc=MFCC_FEATURES):
    """Extract MFCC features from audio signal."""
    try:
        mfccs_mean = compute_mfcc(y, sr, n_mfcc)
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        return None


def load_model(input_features=40):
    """Load the trained model."""
    print(f"Loading model from: {MODEL_PATH}")
    model = DeepfakeCNN(input_features)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
    return model


def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability <= 0.40:
        return "LOW"
    elif probability <= 0.70:
        return "MEDIUM"
    else:
        return "HIGH"


def get_confidence(probability):
    """Determine confidence level based on probability."""
    if probability > 0.80:
        return "High Confidence"
    elif probability >= 0.60:
        return "Moderate Confidence"
    else:
        return "Low Confidence"


def predict(model, audio_path):
    """Make prediction on a single audio file."""
    # Load audio
    print(f"\nLoading audio: {audio_path}")
    y, sr = load_wav_file(audio_path)
    
    if y is None or sr is None:
        return None
    
    # Extract MFCC features
    mfcc_features = extract_mfcc_features(y, sr)
    
    if mfcc_features is None:
        print("Failed to extract features")
        return None
    
    # Reshape for model: (1, features, 1)
    mfcc_tensor = torch.FloatTensor(mfcc_features.reshape(1, MFCC_FEATURES, 1))
    
    # Run inference
    with torch.no_grad():
        output = model(mfcc_tensor)
        probability = torch.sigmoid(output).item()
    
    return probability


def print_result(probability, file_path):
    """Print the prediction result."""
    percentage = probability * 100
    predicted_class = "FAKE" if probability > 0.5 else "REAL"
    risk_level = get_risk_level(probability)
    confidence = get_confidence(probability)
    
    print("-" * 42)
    print(f"Deepfake Probability: {percentage:.1f}%")
    print(f"Prediction: {predicted_class}")
    print(f"Risk Level: {risk_level}")
    print(f"Confidence: {confidence}")
    print("-" * 42)
    
    # Append to log file
    log_entry = f"{file_path} | {percentage:.1f}% | {predicted_class}\n"
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry)
        print(f"\nResult logged to: {LOG_FILE}")
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")


def main():
    """Main entry point."""
    print("=" * 42)
    print("AI Voice Deepfake Detection - Prediction")
    print("=" * 42)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_model.py first to train the model.")
        sys.exit(1)
    
    # Load model
    model = load_model()
    
    # Get audio file path from command line argument
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <path_to_wav_file>")
        print("\nExample:")
        print("  python src/predict.py dataset/real/biden-original.wav")
        print("  python src/predict.py dataset/fake/biden-to-obama.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Make prediction
    probability = predict(model, audio_path)
    
    if probability is not None:
        print_result(probability, audio_path)
    else:
        print("Prediction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
