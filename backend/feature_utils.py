import numpy as np
import librosa
import scipy.signal as sig

# Settings from reference code
SAMPLE_RATE = 16000
LOWPASS_CUTOFF = 4.0
POLY_DEGREE = 5
MAX_FREQ_HZ = 3.0 # 180 BPM
BRV_PEAK_MIN_HEIGHT_FACTOR = 0.5 

def envelope(signal):
    analytic = sig.hilbert(signal)
    return np.abs(analytic)

def lowpass(signal, sr, cutoff=LOWPASS_CUTOFF):
    nyquist = sr / 2
    if cutoff >= nyquist: return signal
    b, a = sig.butter(4, cutoff / nyquist, btype='low')
    return sig.filtfilt(b, a, signal)

def extract_time_domain(y):
    rms = np.sqrt(np.mean(y**2))
    ste = np.mean(y**2)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / 2
    zcr = zero_crossings / len(y)
    return {'RMS': rms, 'STE': ste, 'ZCR': zcr}

def extract_spectral(y, sr):
    S, _ = librosa.magphase(librosa.stft(y))
    cent = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    bw = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    roll = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85))
    flat = np.mean(librosa.feature.spectral_flatness(S=S))
    
    # Peak Freq
    mean_spectrum = np.mean(S, axis=1)
    fft_freqs = librosa.fft_frequencies(sr=sr)
    peak_freq = fft_freqs[np.argmax(mean_spectrum)]
    
    return {
        'Spectral_Centroid': cent, 
        'Spectral_Bandwidth': bw, 
        'Spectral_Rolloff': roll, 
        'Spectral_Flatness': flat,
        'Peak_Freq': peak_freq
    }

def extract_f0(y, sr):
    # F0 using pyin
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=2000, sr=sr)
    valid_f0 = f0[~np.isnan(f0)]
    return np.mean(valid_f0) if len(valid_f0) > 0 else 0.0

def extract_f1(y, sr):
    # F1 using LPC
    try:
        order = 2 + int(sr / 1000)
        a = librosa.lpc(y, order=order)
        rts = np.roots(a)
        rts = [r for r in rts if np.imag(r) >= 0]
        angz = np.arctan2(np.imag(rts), np.real(rts))
        frqs = sorted(angz * (sr / (2 * np.pi)))
        frqs = [f for f in frqs if f > 50]
        return frqs[0] if len(frqs) > 0 else 0.0
    except:
        return 0.0

def extract_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfccs)
    
    feat_dict = {}
    for i, m in enumerate(np.mean(mfccs, axis=1)):
        feat_dict[f'MFCC_{i}'] = m
    for i, d in enumerate(np.mean(delta_mfcc, axis=1)):
        feat_dict[f'Delta_MFCC_{i}'] = d
        
    return feat_dict

def extract_bbi_acf(y, sr):
    # Envelope Analysis for BBI and ACF
    env = envelope(y)
    env_lp = lowpass(env, sr)
    
    # Detrend
    x = np.arange(len(env_lp))
    try:
        p = np.polyfit(x, env_lp, POLY_DEGREE)
        base = np.polyval(p, x)
    except:
        base = np.mean(env_lp)
    norm_env = env_lp - base
    
    # ACF
    acorr = sig.correlate(norm_env, norm_env, mode='full')
    acorr = acorr[len(acorr)//2:]
    acf_conf = 0.0
    if acorr[0] > 0:
        peaks, _ = sig.find_peaks(acorr)
        if len(peaks) > 0:
            acf_conf = np.max(acorr[peaks]) / acorr[0]
            
    # BBI
    peak_min = np.max(norm_env) * BRV_PEAK_MIN_HEIGHT_FACTOR
    min_dist = int(sr / MAX_FREQ_HZ)
    peaks, _ = sig.find_peaks(norm_env, height=peak_min, distance=min_dist)
    
    mean_bbi = 0.0
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / sr
        mean_bbi = np.mean(intervals)
        
    return {'BBI': mean_bbi, 'ACF_Confidence': acf_conf}

def extract_all_features(y, sr=16000):
    feats = {}
    feats.update(extract_time_domain(y))
    feats.update(extract_spectral(y, sr))
    feats['F0'] = extract_f0(y, sr)
    feats['F1'] = extract_f1(y, sr)
    feats.update(extract_mfcc(y, sr))
    feats.update(extract_bbi_acf(y, sr))
    return feats

