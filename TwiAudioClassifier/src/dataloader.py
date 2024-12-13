"""
Dataloader für Twi Audio Klassifizierung.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Union, List, Any
import yaml
import logging
import warnings

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Custom collate function to handle different feature types"""
    audio_tensors = []
    labels = []
    paths = []
    
    for sample in batch:
        audio_tensors.append(sample['audio'])
        labels.append(sample['label'])
        paths.append(sample['path'])
    
    # Stack tensors
    audio_batch = torch.stack(audio_tensors)
    label_batch = torch.stack(labels)
    
    # For CNN input, ensure 4D shape (batch, channel, freq, time)
    if audio_batch.dim() == 5:  # If shape is [B, 1, 1, F, T]
        audio_batch = audio_batch.squeeze(2)  # Remove extra dimension
    
    return {
        'audio': audio_batch,
        'label': label_batch,
        'path': paths
    }

class AudioDataset(Dataset):
    """Dataset für Audio Klassifizierung"""
    
    def __init__(
            self,
            data_file: str,
            config: Dict[str, Any],
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        ):
        """
        Args:
            data_file: Pfad zur TSV Datei
            config: Konfigurations-Dictionary
            transform: Optional Callable für Audio-Transformationen
        """
        self.data = pd.read_csv(data_file, sep='\t')
        self.config = config
        self.transform = transform
        
        # Feature-Statistiken laden falls vorhanden
        self.feature_stats = {}
        if 'stats_file' in config:
            with open(config['stats_file'], 'r') as f:
                self.feature_stats = yaml.safe_load(f)
        
        # Label Encoding
        self.label_to_idx = {
            label: idx for idx, label in enumerate(
                self.data['label'].unique()
            )
        }
        
        # Feature Extraction Setup
        if config.get('feature_type') == 'melspec':
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config['sample_rate'],
                n_fft=config.get('n_fft', 2048),
                hop_length=config.get('hop_length', 512),
                n_mels=config.get('n_mels', 128)
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        elif config.get('feature_type') == 'mfcc':
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=config['sample_rate'],
                n_mfcc=config.get('n_mfcc', 40),
                melkwargs={
                    'n_fft': config.get('n_fft', 2048),
                    'hop_length': config.get('hop_length', 512),
                    'n_mels': config.get('n_mels', 128)
                }
            )
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Features normalisieren"""
        feature_type = self.config.get('feature_type')
        if feature_type in self.feature_stats:
            mean = torch.tensor(self.feature_stats[feature_type]['mean'])
            std = torch.tensor(self.feature_stats[feature_type]['std'])
            
            # Ensure tensors are on the same device
            mean = mean.to(features.device)
            std = std.to(features.device)
            
            # Normalize with broadcasting
            features = (features - mean) / (std + 1e-6)  # Add epsilon for numerical stability
        else:
            warnings.warn(f"Keine Statistiken für {feature_type} gefunden. Verwende Standard-Werte.")
            features = (features - features.mean()) / (features.std() + 1e-6)
        return features
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Feature Extraktion durchführen"""
        if self.config['feature_type'] == 'melspec':
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config['sample_rate'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length'],
                n_mels=self.config['n_mels']
            )
            
            # Log-Mel Spektrogramm berechnen
            mel_spect = mel_spectrogram(waveform)
            mel_spect_db = torchaudio.transforms.AmplitudeToDB()(mel_spect)
            
            # Normalisierung
            if hasattr(self, 'feature_stats'):
                mel_spect_db = self.normalize_features(mel_spect_db)
            
            # Reshape für CNN: [batch_size, channels, height, width]
            return mel_spect_db.unsqueeze(0) if len(mel_spect_db.shape) == 2 else mel_spect_db
            
        elif self.config['feature_type'] == 'mfcc':
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.config['sample_rate'],
                n_mfcc=40,  # Ensure we get 40 coefficients for LSTM
                melkwargs={
                    'n_fft': self.config['n_fft'],
                    'hop_length': self.config['hop_length'],
                    'n_mels': self.config['n_mels']
                }
            )
            
            # MFCC berechnen
            mfcc = mfcc_transform(waveform)  # Shape: [1, n_mfcc, time]
            
            # Normalisierung
            if hasattr(self, 'feature_stats'):
                mfcc = self.normalize_features(mfcc)
            
            # Reshape für LSTM: [batch_size, sequence_length, input_size]
            if len(mfcc.shape) == 3:
                # Permute dimensions to get [batch, time, features]
                mfcc = mfcc.squeeze(0).transpose(0, 1)  # [n_mfcc, time] -> [time, n_mfcc]
            
            # Add batch dimension if needed
            if len(mfcc.shape) == 2:
                mfcc = mfcc.unsqueeze(0)
                
            return mfcc
            
        else:
            raise ValueError(f"Unbekannter Feature-Typ: {self.config['feature_type']}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Ein Audio Sample laden"""
        # Audio laden
        audio_path = str(Path(self.config['data_dir']) / self.data.iloc[idx]['path'])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resampling falls nötig
        if sample_rate != self.config['sample_rate']:
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                self.config['sample_rate']
            )
            waveform = resampler(waveform)
        
        # Mono konvertierung falls nötig
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Länge anpassen falls duration spezifiziert
        target_length = int(self.config['sample_rate'] * self.config.get('duration', 1.0))
        current_length = waveform.shape[1]
        
        if current_length > target_length:
            # Zufälliger Ausschnitt
            start = torch.randint(0, current_length - target_length, (1,))
            waveform = waveform[:, start:start + target_length]
        else:
            # Zero-Padding
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(
                waveform, (0, padding), mode='constant'
            )
        
        # Label laden
        label = self.data.iloc[idx]['label']
        label = torch.tensor(self.label_to_idx[label])
        
        # Features extrahieren
        features = self.extract_features(waveform)
        
        # Optional transform
        if self.transform:
            features = self.transform(features)
        
        return {
            'audio': features,
            'label': label,
            'path': self.data.iloc[idx]['path']
        }

def get_dataloader(
        data_file: str,
        config: Dict[str, Any],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Dict[str, int]]:
    """
    DataLoader erstellen
    
    Args:
        data_file: Pfad zur TSV Datei
        config: Konfigurations-Dictionary
        transform: Optional Callable für Audio-Transformationen
        shuffle: Ob die Daten gemischt werden sollen
    
    Returns:
        DataLoader und Label-Mapping
    """
    dataset = AudioDataset(data_file, config, transform)
    
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    return loader, dataset.label_to_idx
