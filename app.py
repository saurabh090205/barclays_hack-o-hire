"""
AI Voice Deepfake Detection Engine - Streamlit UI
This is the web interface for the AI Voice Deepfake Detection system.
"""

import os
import sys
import wave
import numpy as np
import streamlit as st
import torch

# Add src to path for model import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.model_definition import DeepfakeCNN

# Configuration
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'deepfake_model.pth')
MFCC_FEATURES = 40


def load_model():
    """Load the trained model."""
    model = DeepfakeCNN(MFCC_FEATURES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model


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
        st.error(f"Error loading audio: {e}")
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
        st.error(f"Error computing MFCC: {e}")
        return None


def extract_mfcc_features(y, sr, n_mfcc=MFCC_FEATURES):
    """Extract MFCC features from audio signal."""
    try:
        mfccs_mean = compute_mfcc(y, sr, n_mfcc)
        return mfccs_mean
    except Exception as e:
        st.error(f"Error extracting MFCC features: {e}")
        return None


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


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="AI Voice Deepfake Detection Engine",
        page_icon="🎤",
        layout="centered"
    )
    
    st.title("🎤 AI Voice Deepfake Detection Engine")
    st.markdown("---")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return
    
    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    st.markdown("### Upload Audio File")
    st.info("Please upload a .wav audio file for deepfake detection.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if uploaded_file is not None:
        st.markdown("---")
        
        # Save uploaded file temporarily
        with wave.open(uploaded_file, 'wb') as temp_wav:
            temp_wav.write(uploaded_file.getvalue())
        
        try:
            # Load and process audio
            y, sr = load_wav_file(uploaded_file)
            
            if y is not None:
                st.audio(uploaded_file, format='audio/wav')
                
                # Extract features
                mfcc_features = extract_mfcc_features(y, sr)
                
                if mfcc_features is not None:
                    # Run inference
                    mfcc_tensor = torch.FloatTensor(mfcc_features.reshape(1, MFCC_FEATURES, 1))
                    
                    with torch.no_grad():
                        output = model(mfcc_tensor)
                        probability = torch.sigmoid(output).item()
                    
                    # Display results
                    percentage = probability * 100
                    risk_level = get_risk_level(probability)
                    confidence = get_confidence(probability)
                    
                    st.markdown("### Detection Results")
                    
                    # Probability
                    st.metric(label="Deepfake Probability", value=f"{percentage:.1f}%")
                    
                    # Risk Level with colored display
                    if risk_level == "LOW":
                        st.success(f"🟢 Risk Level: {risk_level}")
                    elif risk_level == "MEDIUM":
                        st.warning(f"🟡 Risk Level: {risk_level}")
                    else:
                        st.error(f"🔴 Risk Level: {risk_level}")
                    
                    # Confidence
                    st.info(f"Confidence: {confidence}")
                    
                    # Additional info
                    with st.expander("See detailed analysis"):
                        st.write(f"**Prediction:** {'FAKE' if probability > 0.5 else 'REAL'}")
                        st.write(f"**Probability (0-1):** {probability:.4f}")
                        st.write(f"**MFCC Features Extracted:** {len(mfcc_features)}")
                        
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        
        # Clean up
        os.remove(uploaded_file.name)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>AI Voice Deepfake Detection Engine | PyTorch 1D CNN</p>
        <p>Model Accuracy: ~92%</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
