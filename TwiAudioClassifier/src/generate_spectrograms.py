import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import logging

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

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "test_config.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "experiments" / "spectrograms"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example audio files for spectrograms (one correct, one incorrect)
    audio_dir = Path(config['data']['data_dir'])
    correct_audio = audio_dir / "common_voice_tw_36831072.mp3"  # Label 1
    incorrect_audio = audio_dir / "common_voice_tw_34745954.mp3"  # Label 0
    
    try:
        # Generate spectrograms
        generate_spectrogram(str(correct_audio), "Correct_Classification", output_dir)
        generate_spectrogram(str(incorrect_audio), "Incorrect_Classification", output_dir)
        logging.info("Generated spectrograms for example classifications")
    except Exception as e:
        logging.error(f"Error generating spectrograms: {e}")

if __name__ == "__main__":
    main()
