from typing import List, Optional
import numpy as np
import pandas as pd
import librosa
import os

# Default feature names in the exact order the model was trained with
DEFAULT_FEATURE_NAMES = [
    # MFCCs
    "MFCC_1","MFCC_2","MFCC_3","MFCC_4","MFCC_5","MFCC_6","MFCC_7",
    "MFCC_8","MFCC_9","MFCC_10","MFCC_11","MFCC_12","MFCC_13",
    # Chroma
    "Chroma_1","Chroma_2","Chroma_3","Chroma_4","Chroma_5","Chroma_6",
    "Chroma_7","Chroma_8","Chroma_9","Chroma_10","Chroma_11","Chroma_12",
    # Spectral Contrast
    "SpectralContrast_1","SpectralContrast_2","SpectralContrast_3",
    "SpectralContrast_4","SpectralContrast_5","SpectralContrast_6","SpectralContrast_7",
    # Other features
    "ZeroCrossingRate","SpectralCentroid"
]

def _safe_load_audio(path: str, sr: int = 22050):
    ## Load audio with librosa, raise informative errors on failure.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        y, sr_ret = librosa.load(path, sr=sr, mono=True)
        if y.size == 0:
            raise ValueError("Loaded audio is empty.")
        return y, sr_ret
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with librosa: {e}")

def _compute_mfcc_means(y: np.ndarray, sr: int, n_mfcc: int = 13) -> List[float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    ## mfcc shape: (n_mfcc, t). Take mean along time axis
    return list(np.mean(mfcc, axis=1))

def _compute_chroma_means(y: np.ndarray, sr: int, n_chroma: int = 12) -> List[float]:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    return list(np.mean(chroma, axis=1))

def _compute_spectral_contrast_means(y: np.ndarray, sr: int, n_bands: int = 6) -> List[float]:
    ## librosa.feature.spectral_contrast returns n_bands+1 rows by time columns.
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands)
    return list(np.mean(contrast, axis=1))  # should return 7 values when n_bands=6

def _compute_zcr_mean(y: np.ndarray) -> float:
    zcr = librosa.feature.zero_crossing_rate(y)
    return float(np.mean(zcr))

def _compute_spectral_centroid_mean(y: np.ndarray, sr: int) -> float:
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(sc))

def extract_features_vector(path: str, sr: int = 22050) -> List[float]:
    """
    Extracts features and returns a list of values in the order:
    MFCC_1..13, Chroma_1..12, SpectralContrast_1..7, ZeroCrossingRate, SpectralCentroid
    """
    y, sr = _safe_load_audio(path, sr=sr)

    mfcc_means = _compute_mfcc_means(y, sr, n_mfcc=13)            # 13
    chroma_means = _compute_chroma_means(y, sr, n_chroma=12)     # 12
    contrast_means = _compute_spectral_contrast_means(y, sr, n_bands=6)  # 7
    zcr_mean = _compute_zcr_mean(y)                              # 1
    sc_mean = _compute_spectral_centroid_mean(y, sr)             # 1

    features = mfcc_means + chroma_means + contrast_means + [zcr_mean, sc_mean]
    if len(features) != len(DEFAULT_FEATURE_NAMES):
        raise ValueError(f"Feature vector length mismatch: expected {len(DEFAULT_FEATURE_NAMES)}, got {len(features)}")
    return features

def extract_features_row(path: str, feature_names: Optional[List[str]] = None, sr: int = 22050) -> pd.DataFrame:
    """
    Returns a single-row pandas DataFrame with columns matching (feature_names or DEFAULT_FEATURE_NAMES).
    If feature_names is provided, the function will reorder/align to it, and fill any missing columns with 0.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES.copy()

    ## Sanity: drop 'class' if user passed full columns including target
    feature_names = [f for f in feature_names if f != "class"]

    vec = extract_features_vector(path, sr=sr)
    df_row = pd.DataFrame([vec], columns=DEFAULT_FEATURE_NAMES)

    ## Align to requested feature_names order. If some names missing in DEFAULT_FEATURE_NAMES, fill with 0.
    aligned = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in df_row.columns:
            aligned[col] = df_row[col]
        else:
            ## Fill missing columns with zeros to avoid breaking the model pipeline
            aligned[col] = 0.0

    ## Ensure dtype numeric
    aligned = aligned.astype(float)
    return aligned
