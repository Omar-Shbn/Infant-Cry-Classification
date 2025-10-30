"""
Infant Cry Classification - Complete Audio Processing Pipeline with SMOTE
Updated version with:
- Overfitting prevention through regularization
- Fixed error handling for numeric labels
- Enhanced model evaluation
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as scipy_signal
import cv2
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# SMOTE for handling imbalance
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'sampling_rate': 22050,
    'signal_duration': 5,  # seconds
    'segment_length': 1024,
    'stride_length': 512,
    'target_frames': 216,
    'n_mfcc': 20,
    'n_mels': 128,
    'fft_window': 1024,
    'hop_length': 256,
    'bandpass_low': 300,  # Hz
    'bandpass_high': 600,  # Hz
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 10
}

print("="*80)
print("INFANT CRY CLASSIFICATION PIPELINE - UPDATED VERSION")
print("="*80)
print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


class AudioFeatureExtractor:
    """
    Extract features from audio signals following the methodology:
    - Time Domain: ZCR, RMS
    - Frequency/Time-Frequency: MFCCs
    - Visualization: Mel-Spectrograms, Time Series Imaging
    """
    
    def __init__(self, config):
        self.config = config
        self.sr = config['sampling_rate']
        self.duration = config['signal_duration']
        self.segment_length = config['segment_length']
        self.stride_length = config['stride_length']
        self.n_mfcc = config['n_mfcc']
        
    def standardize_audio(self, audio_path):
        """
        I. Data Preprocessing - Step 3: Signal Standardization
        Load and standardize audio to 5 seconds at 22,050 Hz
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            # Ensure exactly 5 seconds
            target_length = self.sr * self.duration
            if len(y) < target_length:
                # Pad if shorter
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            elif len(y) > target_length:
                # Trim if longer
                y = y[:target_length]
            
            return y
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def apply_bandpass_filter(self, y):
        """
        Apply band-pass filter (300 Hz - 600 Hz) as specified for infant signals
        """
        nyquist = self.sr / 2
        low = self.config['bandpass_low'] / nyquist
        high = self.config['bandpass_high'] / nyquist
        
        # Design Butterworth bandpass filter
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        y_filtered = scipy_signal.filtfilt(b, a, y)
        
        return y_filtered
    
    def segment_audio(self, y):
        """
        I. Data Preprocessing - Step 4: Signal Segmentation
        Process audio with segment length 1024, stride 512
        """
        segments = []
        for i in range(0, len(y) - self.segment_length + 1, self.stride_length):
            segment = y[i:i + self.segment_length]
            segments.append(segment)
        
        return np.array(segments)
    
    def extract_zcr(self, y):
        """
        II. Feature Extraction - Time Domain: Zero-Crossing Rate
        Measures frequency of changes in signal polarity
        """
        zcr = librosa.feature.zero_crossing_rate(
            y, 
            frame_length=self.segment_length,
            hop_length=self.stride_length
        )
        
        # Pad to 216 frames
        if zcr.shape[1] < self.config['target_frames']:
            padding = self.config['target_frames'] - zcr.shape[1]
            zcr = np.pad(zcr, ((0, 0), (0, padding)), mode='constant')
        
        return zcr.flatten()[:self.config['target_frames']]
    
    def extract_rms(self, y):
        """
        II. Feature Extraction - Time Domain: Root Mean Square
        Measures mean energy of sound signal
        """
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.segment_length,
            hop_length=self.stride_length
        )
        
        # Pad to 216 frames
        if rms.shape[1] < self.config['target_frames']:
            padding = self.config['target_frames'] - rms.shape[1]
            rms = np.pad(rms, ((0, 0), (0, padding)), mode='constant')
        
        return rms.flatten()[:self.config['target_frames']]
    
    def extract_mfcc(self, y):
        """
        II. Feature Extraction - Frequency/Time-Frequency: MFCCs
        Extract 20 MFCC features using 20 Mel bands, FFT window 1024
        Process: pre-emphasis → segmentation → windowing → DFT → 
                Mel-filter → log → DCT
        """
        # Apply pre-emphasis
        pre_emphasis = 0.97
        y_emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y_emphasized,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.config['fft_window'],
            hop_length=self.stride_length,
            n_mels=self.n_mfcc
        )
        
        # Pad to 216 frames
        if mfccs.shape[1] < self.config['target_frames']:
            padding = self.config['target_frames'] - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
        
        # Truncate to 216 frames
        mfccs = mfccs[:, :self.config['target_frames']]
        
        return mfccs
    
    def create_mel_spectrogram(self, y):
        """
        II. Feature Extraction - Visualization: Mel-Spectrograms
        Create 216x216 image: 5-second signal, 22,050 Hz, 128 overlaps, 256 FFT
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=256,
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels'],
            win_length=256
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to 216x216
        mel_spec_resized = cv2.resize(mel_spec_db, (216, 216))
        
        return mel_spec_resized
    
    def mfcc_to_image(self, mfccs):
        """
        II. Feature Extraction - Visualization: Time Series Imaging
        Transform MFCC features into 216x216 pixel images
        """
        # Resize MFCC matrix to 216x216
        mfcc_image = cv2.resize(mfccs, (216, 216))
        
        return mfcc_image
    
    def extract_all_features(self, audio_path, apply_filter=True):
        """
        Extract all features from an audio file
        """
        # Load and standardize
        y = self.standardize_audio(audio_path)
        if y is None:
            return None
        
        # Apply bandpass filter if specified
        if apply_filter:
            y = self.apply_bandpass_filter(y)
        
        # Extract time domain features
        zcr = self.extract_zcr(y)
        rms = self.extract_rms(y)
        
        # Extract frequency domain features
        mfccs = self.extract_mfcc(y)
        
        # Flatten MFCCs for feature vector
        mfcc_features = mfccs.flatten()
        
        # Statistical features from MFCCs
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        
        # Statistical features from ZCR and RMS
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Combine all features
        features = np.concatenate([
            [zcr_mean, zcr_std, rms_mean, rms_std],  # Time domain stats
            mfcc_mean,  # MFCC means
            mfcc_std,   # MFCC stds
            mfcc_max,   # MFCC maxs
            mfcc_min    # MFCC mins
        ])
        
        return {
            'features': features,
            'zcr_sequence': zcr,
            'rms_sequence': rms,
            'mfcc_matrix': mfccs,
            'audio_signal': y
        }


def process_audio_dataset(audio_folder, labels_df, extractor):
    """
    Process entire audio dataset and extract features
    """
    print("\n" + "="*80)
    print("PROCESSING AUDIO DATASET")
    print("="*80)
    
    features_list = []
    labels_list = []
    failed_files = []
    
    for idx, row in labels_df.iterrows():
        audio_file = row['Cry_Audio_File']
        label = row['Cry_Reason']
        
        # Construct full path
        audio_path = os.path.join(audio_folder, audio_file)
        
        if not os.path.exists(audio_path):
            failed_files.append(audio_file)
            continue
        
        # Extract features
        result = extractor.extract_all_features(audio_path)
        
        if result is not None:
            features_list.append(result['features'])
            labels_list.append(label)
        else:
            failed_files.append(audio_file)
        
        # Progress
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(labels_df)} files...")
    
    print(f"\n✅ Successfully processed: {len(features_list)} files")
    print(f"⚠️ Failed to process: {len(failed_files)} files")
    
    if failed_files:
        print(f"\nFailed files (first 10): {failed_files[:10]}")
    
    return np.array(features_list), np.array(labels_list)


def prepare_data_with_smote(X, y, test_size=0.2, random_state=42):
    """
    I. Data Preprocessing - Step 5: Data Split (80% train, 20% test)
    Then apply SMOTE to training data
    """
    print("\n" + "="*80)
    print("DATA PREPARATION AND SMOTE APPLICATION")
    print("="*80)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("\nLabel Encoding:")
    for label, encoded in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {label} → {encoded}")
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_encoded
    )
    
    print(f"\n📊 Data Split (Before SMOTE):")
    print(f"  Training set: {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    print(f"  Testing set: {X_test.shape[0]} samples ({100*test_size:.0f}%)")
    
    print(f"\n📊 Training Set Distribution (Before SMOTE):")
    from collections import Counter
    train_dist = Counter(y_train)
    for label_idx, count in sorted(train_dist.items()):
        print(f"  {le.classes_[label_idx]}: {count} samples")
    
    # Apply SMOTE to training data
    print(f"\n🔄 Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\n📊 Training Set Distribution (After SMOTE):")
    train_dist_balanced = Counter(y_train_balanced)
    for label_idx, count in sorted(train_dist_balanced.items()):
        print(f"  {le.classes_[label_idx]}: {count} samples")
    
    print(f"\n✅ SMOTE Applied:")
    print(f"  Original training samples: {X_train.shape[0]}")
    print(f"  Balanced training samples: {X_train_balanced.shape[0]}")
    print(f"  Synthetic samples created: {X_train_balanced.shape[0] - X_train.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Features standardized using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, le, scaler


def train_models_with_grid_search(X_train, X_test, y_train, y_test, cv_folds=10):
    """
    III. Model Training and Optimization
    Train multiple models with hyperparameter tuning using GridSearchCV
    UPDATED: Added regularization to prevent overfitting
    """
    print("\n" + "="*80)
    print("MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("="*80)
    
    # Define models and their hyperparameter grids
    # UPDATED: More regularization parameters to prevent overfitting
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'Support Vector Classifier': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [5, 10, 15],  # Reduced depth to prevent overfitting
                'min_samples_split': [5, 10, 20],  # Increased to prevent overfitting
                'min_samples_leaf': [2, 4, 8],  # Increased to prevent overfitting
                'max_features': ['sqrt', 'log2']  # Added feature limitation
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],  # Focus on larger forests
                'max_depth': [10, 15, 20],  # Limited depth
                'min_samples_split': [5, 10],  # Higher values
                'min_samples_leaf': [2, 4],  # Higher values
                'max_features': ['sqrt', 'log2'],  # Feature limitation
                'max_samples': [0.7, 0.8, 0.9]  # Bootstrap sample size
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [50, 100],  # Fewer trees
                'max_depth': [3, 5],  # Shallower trees to prevent overfitting
                'learning_rate': [0.01, 0.05, 0.1],  # Slower learning
                'subsample': [0.6, 0.8],  # Use less data per tree
                'colsample_bytree': [0.6, 0.8],  # Use fewer features
                'reg_alpha': [0, 0.1, 1],  # L1 regularization
                'reg_lambda': [1, 5, 10],  # L2 regularization (increased)
                'min_child_weight': [3, 5, 7],  # Prevent overfitting
                'gamma': [0, 0.1, 0.2]  # Minimum loss reduction
            }
        }
    }
    
    results = {}
    
    for name, config in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")
        
        # Perform Grid Search with Cross-Validation
        print(f"Performing Grid Search with {cv_folds}-fold Cross-Validation...")
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        print(f"\n✅ Best Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Cross-validation score
        cv_score = grid_search.best_score_
        print(f"\n📊 Best CV Score (F1-weighted): {cv_score:.4f}")
        
        # Test predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate CV-Test gap for overfitting detection
        cv_test_gap = abs(cv_score - f1)
        
        print(f"\n📈 Test Set Performance:")
        print(f"  Accuracy (ACC):  {accuracy:.4f}")
        print(f"  Precision (PRC): {precision:.4f}")
        print(f"  Recall (REC):    {recall:.4f}")
        print(f"  F1-Score:        {f1:.4f}")
        print(f"\n⚠️  CV-Test Gap:    {cv_test_gap:.4f}", end="")
        
        if cv_test_gap > 0.10:
            print(" 🔴 (High - Possible Overfitting)")
        elif cv_test_gap > 0.05:
            print(" 🟡 (Moderate)")
        else:
            print(" 🟢 (Good Generalization)")
        
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score': cv_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'cv_test_gap': cv_test_gap
        }
    
    return results


def evaluate_and_visualize(results, y_test, label_encoder):
    """
    Evaluate models and create comprehensive visualizations
    UPDATED: Fixed error with numeric labels
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*80)
    
    # Create comparison dataframe
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'CV Score': metrics['cv_score'],
            'Accuracy (ACC)': metrics['accuracy'],
            'Precision (PRC)': metrics['precision'],
            'Recall (REC)': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'CV-Test Gap': metrics['cv_test_gap']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    # Sort by lowest CV-Test gap first (best generalization), then by F1-Score
    comparison_df = comparison_df.sort_values(['CV-Test Gap', 'F1-Score'], ascending=[True, False])
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON (Sorted by Generalization)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Best model (lowest gap + high F1)
    best_model_name = comparison_df.iloc[0]['Model']
    best_metrics = results[best_model_name]
    
    print("\n" + "="*80)
    print("🏆 BEST MODEL (Best Generalization)")
    print("="*80)
    print(f"Model: {best_model_name}")
    print(f"Accuracy (ACC):  {best_metrics['accuracy']:.4f}")
    print(f"Precision (PRC): {best_metrics['precision']:.4f}")
    print(f"Recall (REC):    {best_metrics['recall']:.4f}")
    print(f"F1-Score:        {best_metrics['f1_score']:.4f}")
    print(f"CV Score:        {best_metrics['cv_score']:.4f}")
    print(f"CV-Test Gap:     {best_metrics['cv_test_gap']:.4f}", end="")
    
    if best_metrics['cv_test_gap'] > 0.10:
        print(" 🔴 (Still overfitting - consider more regularization)")
    elif best_metrics['cv_test_gap'] > 0.05:
        print(" 🟡 (Acceptable)")
    else:
        print(" 🟢 (Excellent generalization)")
    
    print(f"\nBest Parameters:")
    for param, value in best_metrics['best_params'].items():
        print(f"  {param}: {value}")
    
    # Classification report - FIXED: Convert labels to strings
    print("\n" + "="*80)
    print(f"CLASSIFICATION REPORT - {best_model_name}")
    print("="*80)
    target_names = [str(label) for label in label_encoder.classes_]
    print(classification_report(y_test, best_metrics['predictions'], 
                                target_names=target_names))
    
    # Visualizations
    create_visualizations(comparison_df, results, y_test, label_encoder, best_model_name)
    
    return comparison_df, best_model_name


def create_visualizations(comparison_df, results, y_test, label_encoder, best_model_name):
    """
    Create comprehensive visualizations
    UPDATED: Fixed confusion matrix labels for numeric classes
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Model Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    metrics_to_plot = ['Accuracy (ACC)', 'Precision (PRC)', 'Recall (REC)', 'F1-Score']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        ax1.bar(x + i*width, comparison_df[metric], width, label=metric)
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: F1-Score Ranking
    ax2 = plt.subplot(2, 3, 2)
    ax2.barh(comparison_df['Model'], comparison_df['F1-Score'], color='skyblue', edgecolor='black')
    ax2.set_xlabel('F1-Score', fontweight='bold')
    ax2.set_title('F1-Score Ranking', fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    for i, v in enumerate(comparison_df['F1-Score']):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 3: CV Score vs Test F1-Score (Overfitting Detection)
    ax3 = plt.subplot(2, 3, 3)
    colors = ['red' if gap > 0.10 else 'orange' if gap > 0.05 else 'green' 
              for gap in comparison_df['CV-Test Gap']]
    ax3.scatter(comparison_df['CV Score'], comparison_df['F1-Score'], 
               s=200, alpha=0.6, c=colors, edgecolors='black')
    for i, model in enumerate(comparison_df['Model']):
        ax3.annotate(model, (comparison_df.iloc[i]['CV Score'], 
                            comparison_df.iloc[i]['F1-Score']),
                    fontsize=8, ha='center')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Generalization')
    ax3.set_xlabel('Cross-Validation Score', fontweight='bold')
    ax3.set_ylabel('Test F1-Score', fontweight='bold')
    ax3.set_title('Overfitting Detection\n(Points near line = good)', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Confusion Matrix for Best Model - FIXED
    ax4 = plt.subplot(2, 3, 4)
    best_predictions = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, best_predictions)
    class_labels = [str(label) for label in label_encoder.classes_]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    ax4.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold', fontsize=12)
    ax4.set_ylabel('True Label', fontweight='bold')
    ax4.set_xlabel('Predicted Label', fontweight='bold')
    
    # Plot 5: CV-Test Gap (Overfitting Measure)
    ax5 = plt.subplot(2, 3, 5)
    colors = ['red' if gap > 0.10 else 'orange' if gap > 0.05 else 'green' 
              for gap in comparison_df['CV-Test Gap']]
    ax5.barh(comparison_df['Model'], comparison_df['CV-Test Gap'], 
            color=colors, edgecolor='black')
    ax5.set_xlabel('CV-Test Gap', fontweight='bold')
    ax5.set_title('Overfitting Measure\n(Lower = Better)', fontweight='bold', fontsize=12)
    ax5.invert_yaxis()
    ax5.axvline(x=0.05, color='orange', linestyle='--', label='Acceptable (0.05)', alpha=0.7)
    ax5.axvline(x=0.10, color='red', linestyle='--', label='High (0.10)', alpha=0.7)
    ax5.legend(fontsize=8)
    for i, v in enumerate(comparison_df['CV-Test Gap']):
        ax5.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 6: Metrics Heatmap
    ax6 = plt.subplot(2, 3, 6)
    metrics_df = comparison_df.set_index('Model')[metrics_to_plot]
    sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax6,
                cbar_kws={'label': 'Score'})
    ax6.set_title('Metrics Heatmap', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Model', fontweight='bold')
    ax6.set_ylabel('Metric', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/comprehensive_model_evaluation.png', dpi=300, bbox_inches='tight')
    print("✅ Comprehensive visualization saved!")
    plt.close()


def main_pipeline_with_csv(csv_path):
    """
    Main pipeline using CSV file with pre-extracted features
    (When audio files are not available)
    """
    print("\n" + "="*80)
    print("RUNNING PIPELINE WITH PRE-EXTRACTED FEATURES")
    print("="*80)
    
    # Load CSV
    print("\n📂 Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} samples")
    
    # Drop audio file column
    if 'Cry_Audio_File' in df.columns:
        df = df.drop('Cry_Audio_File', axis=1)
    
    # Separate features and labels
    X = df.drop('Cry_Reason', axis=1).values
    y = df['Cry_Reason'].values
    
    print(f"\n📊 Dataset Info:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Prepare data with SMOTE
    X_train, X_test, y_train, y_test, label_encoder, scaler = prepare_data_with_smote(
        X, y, 
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    # Train models with grid search
    results = train_models_with_grid_search(
        X_train, X_test, y_train, y_test,
        cv_folds=CONFIG['cv_folds']
    )
    
    # Evaluate and visualize
    comparison_df, best_model_name = evaluate_and_visualize(results, y_test, label_encoder)
    
    # Save results
    comparison_df.to_csv('/mnt/user-data/outputs/model_comparison_results.csv', index=False)
    print("\n✅ Results saved to: model_comparison_results.csv")
    
    # Save best model
    import joblib
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, '/mnt/user-data/outputs/best_model_smote.pkl')
    joblib.dump(scaler, '/mnt/user-data/outputs/scaler_smote.pkl')
    joblib.dump(label_encoder, '/mnt/user-data/outputs/label_encoder_smote.pkl')
    print("\n✅ Models saved!")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    return results, comparison_df, best_model_name


def main_pipeline_with_audio(audio_folder, labels_csv):
    """
    Main pipeline with audio files
    (When raw audio files are available)
    """
    print("\n" + "="*80)
    print("RUNNING COMPLETE AUDIO PROCESSING PIPELINE")
    print("="*80)
    
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor(CONFIG)
    
    # Process audio dataset
    X, y = process_audio_dataset(audio_folder, labels_df, extractor)
    
    # Prepare data with SMOTE
    X_train, X_test, y_train, y_test, label_encoder, scaler = prepare_data_with_smote(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    # Train models with grid search
    results = train_models_with_grid_search(
        X_train, X_test, y_train, y_test,
        cv_folds=CONFIG['cv_folds']
    )
    
    # Evaluate and visualize
    comparison_df, best_model_name = evaluate_and_visualize(results, y_test, label_encoder)
    
    # Save results
    comparison_df.to_csv('/mnt/user-data/outputs/model_comparison_results.csv', index=False)
    print("\n✅ Results saved!")
    
    return results, comparison_df, best_model_name


if __name__ == "__main__":
    print("\n" + "="*80)
    print("INFANT CRY CLASSIFICATION - UPDATED METHODOLOGY")
    print("="*80)
    print("\nUpdates in this version:")
    print("✅ Enhanced regularization to prevent overfitting")
    print("✅ Fixed error with numeric label encoding")
    print("✅ Added overfitting detection metrics")
    print("✅ Improved XGBoost hyperparameters")
    print("✅ Model selection based on generalization")
    print("\nThis script implements:")
    print("1. Signal Standardization (5-sec, 22,050 Hz)")
    print("2. Signal Segmentation (1024 segment, 512 stride, 216 frames)")
    print("3. Feature Extraction (ZCR, RMS, MFCCs)")
    print("4. Data Split (80% train, 20% test)")
    print("5. SMOTE for class balancing")
    print("6. Model Training (LR, SVC, DT, RF, XGBoost)")
    print("7. Hyperparameter Tuning (Grid Search)")
    print("8. Cross-Validation (10-fold)")
    print("9. Evaluation (ACC, PRC, REC, F1)")
    
    print("\n" + "="*80)
    print("CHOOSE MODE:")
    print("="*80)
    print("1. Use with pre-extracted features (CSV file)")
    print("2. Use with raw audio files")
    
    # For this example, use CSV mode
    csv_path = r'C:\Users\ADMIN\Downloads\Cry classification\donateacry-corpus_features_final.csv'
    
    results, comparison_df, best_model = main_pipeline_with_csv(csv_path)