"""
Datenvorbereitung für Audio-Klassifizierung.
Erstellt TSV-Dateien für Training und Test aus rohen Audiodaten.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple
import random
import shutil
import yaml

def prepare_dataset(
    audio_dir: Path,
    output_dir: Path,
    split_ratio: float = 0.2
) -> Tuple[Path, Path]:
    """
    Bereitet Datensatz vor und teilt in Train/Test.
    
    Args:
        audio_dir: Verzeichnis mit Audiodateien
        output_dir: Ausgabeverzeichnis für TSV-Dateien
        split_ratio: Anteil der Testdaten (0-1)
        
    Returns:
        Tuple von (train_file, test_file) Pfaden
    """
    logging.info("Bereite Datensatz vor...")
    
    # Alle WAV-Dateien finden
    audio_files = list(audio_dir.glob('**/*.wav'))
    
    # Daten sammeln
    data = []
    for audio_file in audio_files:
        # Label aus Verzeichnisname
        label = audio_file.parent.name
        if label not in ['twi', 'other']:
            continue
            
        # Label zu 0/1 konvertieren
        label_id = 1 if label == 'twi' else 0
        
        data.append({
            'path': str(audio_file.relative_to(audio_dir)),
            'label': label_id
        })
    
    # Zu DataFrame
    df = pd.DataFrame(data)
    
    # Train/Test Split
    test_size = int(len(df) * split_ratio)
    test_indices = random.sample(range(len(df)), test_size)
    train_indices = list(set(range(len(df))) - set(test_indices))
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    # Ausgabeverzeichnis erstellen
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TSV-Dateien speichern
    train_file = output_dir / 'train.tsv'
    test_file = output_dir / 'test.tsv'
    
    train_df.to_csv(train_file, sep='\t', index=False)
    test_df.to_csv(test_file, sep='\t', index=False)
    
    logging.info(f"Train-Datensatz: {len(train_df)} Samples")
    logging.info(f"Test-Datensatz: {len(test_df)} Samples")
    
    return train_file, test_file

def create_simple_dataset(
    train_file: Path,
    test_file: Path,
    output_dir: Path,
    max_samples: int = 1000
) -> Tuple[Path, Path]:
    """
    Erstellt vereinfachten Datensatz für schnelles Training/Testing.
    
    Args:
        train_file: Ursprüngliche Train-TSV
        test_file: Ursprüngliche Test-TSV
        output_dir: Ausgabeverzeichnis
        max_samples: Maximale Anzahl Samples pro Split
        
    Returns:
        Tuple von (simple_train_file, simple_test_file)
    """
    logging.info("Erstelle vereinfachten Datensatz...")
    
    # TSVs laden
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    
    # Samples zufällig auswählen
    train_df = train_df.sample(n=min(max_samples, len(train_df)))
    test_df = test_df.sample(n=min(max_samples//2, len(test_df)))
    
    # Speichern
    simple_train = output_dir / 'train_simple.tsv'
    simple_test = output_dir / 'test_simple.tsv'
    
    train_df.to_csv(simple_train, sep='\t', index=False)
    test_df.to_csv(simple_test, sep='\t', index=False)
    
    logging.info(f"Vereinfachter Train-Datensatz: {len(train_df)} Samples")
    logging.info(f"Vereinfachter Test-Datensatz: {len(test_df)} Samples")
    
    return simple_train, simple_test

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=Path, required=True,
                      help='Verzeichnis mit Audiodateien')
    parser.add_argument('--output_dir', type=Path, required=True,
                      help='Ausgabeverzeichnis für TSV-Dateien')
    parser.add_argument('--create_simple', action='store_true',
                      help='Erstelle auch vereinfachten Datensatz')
    parser.add_argument('--split_ratio', type=float, default=0.2,
                      help='Anteil der Testdaten (0-1)')
    args = parser.parse_args()
    
    # Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Hauptdatensatz erstellen
    train_file, test_file = prepare_dataset(
        args.audio_dir,
        args.output_dir,
        args.split_ratio
    )
    
    # Optional: Vereinfachten Datensatz erstellen
    if args.create_simple:
        create_simple_dataset(
            train_file,
            test_file,
            args.output_dir
        )

if __name__ == '__main__':
    main()
