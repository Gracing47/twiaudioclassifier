"""
Test Set Evaluation für Twi Audio Klassifizierung.
Evaluiert trainierte Modelle auf dem Testset.
"""

import torch
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import librosa
import librosa.display

from models import get_model, CNNModel
from dataloader import get_dataloader
from trainer import ModelTrainer

def setup_logging(output_dir: Path) -> None:
    """Konfiguriert Logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'test_evaluation.log'),
            logging.StreamHandler()
        ]
    )

def load_model(model_type: str, config: Dict) -> torch.nn.Module:
    """Lädt ein vortrainiertes Modell"""
    
    # Modell initialisieren
    model_config = config['models'][model_type]
    model = get_model(model_type, num_classes=2, config=model_config)
    
    # Checkpoint laden
    checkpoint_path = config['checkpoints'][model_type]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Fehler beim Laden des Modells: {str(e)}")
        raise e

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float, List[int], List[int], List[float], float, Dict]:
    """
    Evaluiert Modell auf Testset
    
    Args:
        model: Trainiertes Modell
        test_loader: DataLoader für Testdaten
        device: GPU oder CPU
        
    Returns:
        Accuracy, Loss, Predictions, Ground Truth, Confidences, AUC-ROC, Additional Metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0], device=device))
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_probs = []
    all_raw_outputs = []  # Store raw logits
    
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Feature statistics
                logging.info(f"\nBatch {batch_idx}:")
                logging.info(f"Audio shape: {audio.shape}")
                logging.info(f"Audio min/max: {audio.min():.4f}/{audio.max():.4f}")
                logging.info(f"Audio mean/std: {audio.mean():.4f}/{audio.std():.4f}")
                
                # Forward pass
                outputs = model(audio)
                all_raw_outputs.extend(outputs.cpu().numpy())
                
                # Loss calculation
                loss = criterion(outputs, labels)
                
                # Predictions and probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1, keepdim=True)
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if preds[i] == labels[i]:
                        class_correct[label] += 1
                
                # Logging
                logging.info(f"Raw outputs: {outputs}")
                logging.info(f"Probabilities: {probabilities}")
                logging.info(f"Predictions: {preds.squeeze()}")
                logging.info(f"True labels: {labels}")
                
                # Metrics
                total_loss += loss.item()
                correct += preds.eq(labels.view_as(preds)).sum().item()
                total += labels.size(0)
                
                confidences = probabilities.max(dim=1)[0]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy())
                
            except Exception as e:
                logging.error(f"Fehler bei der Verarbeitung eines Batches: {e}")
                continue
    
    # Calculate metrics
    accuracy = 100. * correct / total if total > 0 else 0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    
    # Per-class accuracy
    class_accuracies = {
        cls: (class_correct[cls] / class_total[cls] * 100 if class_total[cls] > 0 else 0)
        for cls in [0, 1]
    }
    
    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        logging.error(f"Fehler bei der AUC-ROC Berechnung: {e}")
        auc_roc = 0.0
    
    # Additional metrics
    additional_metrics = {
        'class_accuracies': class_accuracies,
        'raw_output_stats': {
            'mean': np.mean(all_raw_outputs, axis=0),
            'std': np.std(all_raw_outputs, axis=0),
            'min': np.min(all_raw_outputs, axis=0),
            'max': np.max(all_raw_outputs, axis=0)
        },
        'confidence_stats': {
            'mean': np.mean(all_confidences),
            'std': np.std(all_confidences),
            'min': np.min(all_confidences),
            'max': np.max(all_confidences)
        }
    }
    
    return accuracy, avg_loss, all_preds, all_labels, all_confidences, auc_roc, additional_metrics

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], output_dir: Path) -> None:
    """
    Erstellt Confusion Matrix Plot
    
    Args:
        y_true: Ground Truth Labels
        y_pred: Vorhergesagte Labels
        output_dir: Ausgabeverzeichnis
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
    except Exception as e:
        logging.error(f"Fehler beim Erstellen der Confusion Matrix: {e}")

def plot_confidence_distribution(confidences: List[float], output_dir: Path) -> None:
    """
    Erstellt Verteilungsplot der Vorhersage-Konfidenz
    
    Args:
        confidences: Liste der Konfidenzwerte
        output_dir: Ausgabeverzeichnis
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, edgecolor='black')
        plt.title('Verteilung der Vorhersage-Konfidenz')
        plt.xlabel('Konfidenz')
        plt.ylabel('Anzahl')
        plt.savefig(output_dir / 'confidence_distribution.png')
        plt.close()
    except Exception as e:
        logging.error(f"Fehler beim Erstellen des Konfidenz-Plots: {e}")

def generate_spectrogram(audio_path, title, output_dir):
    """Generate and save a spectrogram from an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        title (str): Title for the spectrogram plot
        output_dir (Path): Directory to save the spectrogram
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=32000)  # Use the same sample rate as in config
    # Generate spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / f"{title.replace(' ', '_')}.png"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved spectrogram to {output_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Twi Audio Classifier Test Evaluation')
    parser.add_argument('--config', type=str, default='config/test_config.yml',
                      help='Pfad zur Test-Konfigurationsdatei')
    parser.add_argument('--model', type=str, choices=['cnn', 'lstm', 'all'],
                      default='all', help='Zu evaluierendes Modell')
    return parser.parse_args()

def main():
    """Hauptfunktion"""
    # Parse arguments
    args = parse_args()
    
    try:
        # Lade Test-Konfiguration
        with open(args.config, 'r') as f:
            test_config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Fehler beim Laden der Konfiguration: {e}")
        sys.exit(1)
    
    # Output Verzeichnis
    output_dir = Path(test_config['evaluation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging Setup
    setup_logging(output_dir)
    
    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Verwende Device: {device}")
    
    # Modelle für Evaluation auswählen
    if args.model == 'all':
        models_to_evaluate = ['cnn', 'lstm']
    else:
        models_to_evaluate = [args.model]
    
    # Evaluierung für jedes ausgewählte Modell
    results = {}
    for model_name in models_to_evaluate:
        model_config = test_config['models'][model_name]
        logging.info(f"\nEvaluiere {model_name.upper()} Modell...")
        
        try:
            # Modell-spezifisches Output-Verzeichnis
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            # Dataloader Konfiguration
            dataloader_config = {
                'data_dir': test_config['data']['data_dir'],
                'sample_rate': test_config['features']['sample_rate'],
                'duration': test_config['features']['target_length'],
                'batch_size': test_config['evaluation']['batch_size'],
                'num_workers': test_config['evaluation']['num_workers'],
                'feature_type': model_config['feature_type'],
                'n_fft': test_config['features']['n_fft'],
                'hop_length': test_config['features']['hop_length'],
                'n_mels': test_config['features']['n_mels'],
                'stats_file': test_config['features']['stats_file']
            }
            
            # Test DataLoader
            test_loader, label_mapping = get_dataloader(
                data_file=test_config['data']['test_metadata'],
                config=dataloader_config,
                shuffle=False  # Keine Mischung für Testdaten
            )
            
            # Modell laden
            model = load_model(model_name, test_config)
            
            # Evaluierung
            accuracy, loss, predictions, labels, confidences, auc_roc, additional_metrics = evaluate_model(
                model,
                test_loader,
                device
            )
            
            # Metriken speichern
            results[model_name] = {
                'accuracy': accuracy,
                'loss': loss,
                'predictions': predictions,
                'labels': labels,
                'confidences': confidences,
                'auc_roc': auc_roc,
                'additional_metrics': additional_metrics
            }
            
            # Visualisierungen
            if 'confusion_matrix' in test_config['evaluation']['visualizations']:
                plot_confusion_matrix(labels, predictions, model_output_dir)
            
            if 'prediction_distribution' in test_config['evaluation']['visualizations']:
                plot_confidence_distribution(confidences, model_output_dir)
            
            # Classification Report
            report = classification_report(labels, predictions)
            with open(model_output_dir / 'classification_report.txt', 'w') as f:
                f.write(str(report))
            
            logging.info(f"{model_name.upper()} Test Accuracy: {accuracy:.2f}%")
            logging.info(f"{model_name.upper()} Test Loss: {loss:.4f}")
            logging.info(f"{model_name.upper()} AUC-ROC: {auc_roc:.4f}")
            logging.info(f"\nClassification Report:\n{report}")
            
            # Per-class metrics
            logging.info('\nPer-class Accuracies:')
            for cls, acc in additional_metrics['class_accuracies'].items():
                logging.info(f'Class {cls}: {acc:.2f}%')
            
            # Raw output statistics
            logging.info('\nRaw Output Statistics:')
            for stat, values in additional_metrics['raw_output_stats'].items():
                logging.info(f'{stat}: {values}')
            
            # Confidence statistics
            logging.info('\nConfidence Statistics:')
            for stat, value in additional_metrics['confidence_stats'].items():
                logging.info(f'{stat}: {value:.4f}')
            
            # Generate spectrograms
            correct_audio = Path(test_config['data']['audio_dir']) / "common_voice_tw_36831072.mp3"  # Label 1
            incorrect_audio = Path(test_config['data']['audio_dir']) / "common_voice_tw_34745954.mp3"  # Label 0
            generate_spectrogram(str(correct_audio), "Correct_Classification", model_output_dir)
            generate_spectrogram(str(incorrect_audio), "Incorrect_Classification", model_output_dir)
            logging.info("Generated spectrograms for example classifications")
            
        except Exception as e:
            logging.error(f"Fehler bei der Evaluation von {model_name}: {e}")
            continue
    
    # Vergleichende Analyse
    if len(results) > 1:
        try:
            comparison_results = {
                'Model': [],
                'Accuracy': [],
                'Loss': [],
                'AUC-ROC': []
            }
            
            for model_name, metrics in results.items():
                comparison_results['Model'].append(model_name.upper())
                comparison_results['Accuracy'].append(metrics['accuracy'])
                comparison_results['Loss'].append(metrics['loss'])
                comparison_results['AUC-ROC'].append(metrics['auc_roc'])
            
            # Ergebnisse als DataFrame speichern
            df_results = pd.DataFrame(comparison_results)
            df_results.to_csv(output_dir / 'model_comparison.csv', index=False)
            
            # DataFrame als String formatieren
            results_str = df_results.to_string()
            logging.info("\nVergleichende Ergebnisse:")
            logging.info(results_str)
        except Exception as e:
            logging.error(f"Fehler bei der vergleichenden Analyse: {e}")
    
    # Results in YAML speichern
    with open(output_dir / 'evaluation_results.yml', 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    main()
