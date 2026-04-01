import os
import torch
import numpy as np
import librosa
import xgboost as xgb
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import feature_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AST_MODEL_PATH = "/app/best_ast_model"
XGB_MODEL_PATH = "/app/ultimate_xgb.json"

# Thresholds from 20260302_ultimate config
T_N = 0.70
T_A = 0.25

class AudioProcessor:
    def __init__(self):
        self.ast_path = AST_MODEL_PATH if os.path.exists(AST_MODEL_PATH) else "../20260209_n/best_ast_model"
        self.xgb_path = XGB_MODEL_PATH if os.path.exists(XGB_MODEL_PATH) else "ultimate_xgb.json"
        
        print(f"Loading AST model from {self.ast_path} on {DEVICE}...")
        self.processor = ASTFeatureExtractor.from_pretrained(self.ast_path)
        # Use ASTForAudioClassification as requested
        self.ast_model = ASTForAudioClassification.from_pretrained(
            self.ast_path, 
            output_hidden_states=True, 
            ignore_mismatched_sizes=True
        )
        self.ast_model.to(DEVICE)
        self.ast_model.eval()

        print(f"Loading XGBoost model from {self.xgb_path}...")
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(self.xgb_path)
        
    def predict(self, audio_path: str):
        try:
            yield {"status": "progress", "message": "Loading audio file...", "progress": 0.05}
            # 1. Load entire audio
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            segment_duration = 10.0
            hop = 10.0
            curr = 0.0
            
            all_probs = []
            
            total_segments = max(1, int(np.ceil((duration - segment_duration) / hop)) + 1)
            if duration <= segment_duration:
                total_segments = 1
                
            yield {"status": "progress", "message": f"Segmenting audio into {total_segments} clips...", "progress": 0.1}
            
            segment_idx = 0
            detailed_segments = []
            
            # Segment the audio into 10s clips with 10s hop
            while curr + segment_duration <= max(duration, segment_duration):
                yield {"status": "progress", "message": f"Processing segment {segment_idx+1}/{total_segments}...", "progress": 0.1 + 0.8 * (segment_idx / total_segments)}
                
                start_sample = int(curr * sr)
                end_sample = int((curr + segment_duration) * sr)
                segment = y[start_sample:end_sample]
                
                target_len = int(16000 * segment_duration)
                if len(segment) < target_len:
                    segment = np.pad(segment, (0, target_len - len(segment)))
                else:
                    segment = segment[:target_len]
                    
                # 2. Extract Traditional Features
                trad_dict = feature_utils.extract_all_features(segment, sr=16000)
                trad_feat = np.array(list(trad_dict.values()))
                
                # 3. Extract AST Features
                with torch.no_grad():
                    inputs = self.processor(segment, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
                    # We output hidden states to get the embeddings
                    outputs = self.ast_model(inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-3:] 
                    hs_stack = torch.stack(hidden_states)
                    avg_layers = torch.mean(hs_stack, dim=0)
                    global_pool = torch.mean(avg_layers, dim=1)
                    ast_feat = global_pool.cpu().numpy().squeeze()
                    
                # Combine or use AST only depending on model
                feature_names = self.xgb_model.feature_names
                expected_feats = len(feature_names) if feature_names else 768
                
                if expected_feats == 768:
                    final_feat = ast_feat
                else:
                    final_feat = np.concatenate([ast_feat, trad_feat])
                    final_feat = final_feat[:expected_feats] # Fallback truncation
                
                # Determine correct feature names for the XGBoost predict matrix
                if feature_names is None or len(feature_names) != len(final_feat):
                    feature_names = [f"f{i}" for i in range(len(final_feat))]
                    
                dtest = xgb.DMatrix(final_feat.reshape(1, -1), feature_names=feature_names)
                probs = self.xgb_model.predict(dtest)
                if probs.ndim == 2:
                    probs = probs[0]
                    
                all_probs.append(probs)
                detailed_segments.append({
                    "start_time": curr,
                    "end_time": curr + segment_duration,
                    "probabilities": {
                        "no-breathing": float(probs[0]),
                        "normal breathing": float(probs[1]),
                        "abnormal breathing": float(probs[2])
                    }
                })
                
                if curr + segment_duration >= duration:
                    break
                
                curr += hop
                segment_idx += 1

            yield {"status": "progress", "message": "Aggregating probabilities...", "progress": 0.95}
            # Combine probabilities by averaging across all segments
            mean_probs = np.mean(all_probs, axis=0) if all_probs else np.array([0.0, 0.0, 0.0])
            
            # Classes are typically: 0: No-Breathing, 1: Normal, 2: Abnormal
            # Apply thresholds
            final_pred_idx = 0
            if mean_probs[2] >= T_A: final_pred_idx = 2
            elif mean_probs[1] >= T_N: final_pred_idx = 1
            else: final_pred_idx = 0
            
            classes = ["no-breathing", "normal breathing", "abnormal breathing"]
            
            # Compute a lightweight envelope for the frontend waveform plot (1000 points)
            points = 1000
            chunk_size = max(1, len(y) // points)
            y_trunc = y[:chunk_size * points]
            envelope = np.max(np.abs(y_trunc.reshape(-1, chunk_size)), axis=1).tolist() if len(y_trunc) > 0 else []
            
            # Compute a lightweight mel-spectrogram for the frontend (e.g. 128 bins x 500 time steps)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Downsample spectrogram time dimension for JSON efficiency (limit to ~400 points)
            target_time_bins = 400
            if S_db.shape[1] > target_time_bins:
                hop_s = S_db.shape[1] // target_time_bins
                S_db_small = S_db[:, ::hop_s]
            else:
                S_db_small = S_db
                
            spectrogram = S_db_small.tolist()

            yield {
                "status": "success",
                "probabilities": {
                    "no-breathing": float(mean_probs[0]),
                    "normal breathing": float(mean_probs[1]),
                    "abnormal breathing": float(mean_probs[2])
                },
                "prediction": classes[final_pred_idx],
                "segments_analyzed": len(all_probs),
                "detailed_segments": detailed_segments,
                "envelope": envelope,
                "duration": duration,
                "spectrogram": spectrogram
            }
            
        except Exception as e:
            yield {"status": "error", "message": str(e)}

# Singleton instantiation for the backend
processor_instance = None

def get_processor():
    global processor_instance
    if processor_instance is None:
        processor_instance = AudioProcessor()
    return processor_instance
