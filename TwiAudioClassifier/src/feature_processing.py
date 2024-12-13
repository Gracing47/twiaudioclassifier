"""
Feature-Verarbeitung und Statistik-Berechnung für Audio-Klassifizierung.
"""

import torch
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

from dataloader import AudioDataset

def calculate_feature_statistics(
    data_file: str,
    config: dict,
    feature_type: str
) -> Tuple[float, float]:
    """
    Berechnet Mean und Std der Features aus dem Trainingsdatensatz.
    
    Args:
        data_file: Pfad zur TSV Datei mit Trainingsdaten
        config: Konfigurations-Dictionary
        feature_type: Art der Features ('melspec' oder 'mfcc')
        
    Returns:
        Tuple von (mean, std) der Features
    """
    logging.info(f"Berechne Feature-Statistiken für {feature_type}...")
    
    # Dataset ohne Normalisierung erstellen
    config['feature_type'] = feature_type
    dataset = AudioDataset(data_file, config)
    
    # Features sammeln
    all_features = []
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        features = sample['audio']
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)
        if not features.is_floating_point():
            features = features.to(torch.float32)
        all_features.append(features)
    
    # Zu einem Tensor zusammenfügen
    all_features = torch.cat([f.reshape(-1) for f in all_features])
    
    # Statistiken berechnen
    mean = torch.mean(all_features).item()
    std = torch.std(all_features).item()
    
    logging.info(f"Berechnete Statistiken für {feature_type}:")
    logging.info(f"Mean: {mean:.4f}")
    logging.info(f"Std: {std:.4f}")
    
    return mean, std

def save_statistics(
    stats: Dict[str, Tuple[float, float]],
    output_file: str
) -> None:
    """
    Speichert Feature-Statistiken in YAML-Datei.
    
    Args:
        stats: Dictionary mit Feature-Typ als Key und (mean, std) als Value
        output_file: Pfad zur Output YAML-Datei
    """
    # Statistiken in Dictionary formatieren
    stats_dict = {
        feature_type: {
            'mean': mean,
            'std': std
        }
        for feature_type, (mean, std) in stats.items()
    }
    
    # Als YAML speichern
    with open(output_file, 'w') as f:
        yaml.dump(stats_dict, f)
    
    logging.info(f"Statistiken gespeichert in {output_file}")

def compute_all_statistics():
    """
    Berechnet Statistiken für beide Feature-Typen (MFCC und Mel-Spektrogramm)
    """
    # Basis-Konfiguration
    base_config = {
        'data_dir': 'data/raw/clips',
        'sample_rate': 16000,
        'n_fft': 2048,
        'hop_length': 512
    }
    
    # Statistiken für beide Feature-Typen berechnen
    stats = {}
    
    # CNN (melspec) Statistiken
    mean, std = calculate_feature_statistics(
        data_file='data/processed/train_simple.tsv',
        config={
            **base_config,
            'feature_type': 'melspec',
            'n_mels': 128
        },
        feature_type='melspec'
    )
    stats['melspec'] = (mean, std)
    
    # LSTM (mfcc) Statistiken
    mean, std = calculate_feature_statistics(
        data_file='data/processed/train_simple.tsv',
        config={
            **base_config,
            'feature_type': 'mfcc',
            'n_mfcc': 40
        },
        feature_type='mfcc'
    )
    stats['mfcc'] = (mean, std)
    
    # Statistiken speichern
    output_dir = Path('experiments/feature_stats')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_statistics(stats, str(output_dir / 'feature_stats.yml'))

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'all'], default='all',
                      help='Berechnung für einzelnen Feature-Typ oder alle')
    parser.add_argument('--config', help='Pfad zur Konfigurationsdatei (nur für mode=single)')
    parser.add_argument('--train_file', help='Pfad zur Trainings-TSV (nur für mode=single)')
    parser.add_argument('--feature_type', choices=['melspec', 'mfcc'],
                      help='Feature-Typ (nur für mode=single)')
    args = parser.parse_args()
    
    # Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.mode == 'single':
        if not all([args.config, args.train_file, args.feature_type]):
            parser.error("Für mode=single werden --config, --train_file und --feature_type benötigt")
            
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
        mean, std = calculate_feature_statistics(
            args.train_file,
            config,
            args.feature_type
        )
        
        stats = {args.feature_type: (mean, std)}
        save_statistics(stats, 'experiments/feature_stats/single_stats.yml')
    else:
        compute_all_statistics()

if __name__ == '__main__':
    main()
