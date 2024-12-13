"""
Modell Architekturen für Twi Audio Klassifizierung.
Implementiert CNN und LSTM basierte Modelle mit Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class AttentionLayer(nn.Module):
    """Attention Layer für LSTM"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # Attention Weights berechnen
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        
        # Gewichtete Summe
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_output

class TwiLSTM(nn.Module):
    """LSTM Modell mit Attention"""
    
    def __init__(
            self,
            input_size: int = 40,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.5,
            num_classes: int = 10
        ):
        """
        Args:
            input_size: Größe des Input Features (n_mfcc)
            hidden_size: Größe des Hidden States
            num_layers: Anzahl LSTM Layer
            dropout: Dropout Rate
            num_classes: Anzahl der Klassen
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = AttentionLayer(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass durch das LSTM mit Attention"""
        # Ensure input has shape (batch, time, features)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim
        elif x.dim() == 4:
            # If from CNN features, reshape to (batch, time, features)
            b, c, h, w = x.size()
            x = x.squeeze(1).transpose(1, 2)  # [batch, 1, time, features] -> [batch, time, features]
            
        # LSTM layers
        x = x.float()  # Ensure float type
        output, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(output), dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), output)
        
        # Final prediction
        x = context.squeeze(1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        
        return x

class CNNModel(nn.Module):
    def __init__(
            self,
            channels: List[int] = [32, 64, 128],
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            pool_size: int = 2,
            dropout: float = 0.5,
            dense_units: int = 256,
            num_classes: int = 2
        ):
        super().__init__()
        
        # Convolutional Layers
        layers = []
        in_channels = 1
        
        for out_channels in channels:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                nn.Dropout2d(dropout)
            )
            layers.append(conv_block)
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive Pooling für variable Input Größe
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully Connected Layer
        self.fc_layers = nn.Sequential(
            nn.Linear(channels[-1] * 16, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights for better class balance"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass durch das CNN"""
        # Ensure input has shape (batch, channel, freq, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim
            
        # Conv layers
        x = self.conv_layers(x)
        
        # Global pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc_layers(x)
        
        return x

class LSTMModel(nn.Module):
    def __init__(
            self,
            input_size: int = 40,
            hidden_size: int = 32,  # Will be doubled due to bidirectional
            num_layers: int = 1,
            dropout: float = 0.5,
            num_classes: int = 2
        ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Match saved model
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.ModuleDict({
            'attention': nn.Linear(hidden_size * 2, 1)  # Double size for bidirectional
        })
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Double size for bidirectional
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def attention_net(self, lstm_output):
        """Apply attention mechanism"""
        # lstm_output shape: (batch, seq_len, hidden_size * 2)
        attention_weights = F.softmax(
            self.attention['attention'](lstm_output), 
            dim=1
        )
        # Weighted sum
        context = torch.bmm(
            attention_weights.transpose(1, 2),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_size * 2)
        )
        return context.squeeze(1)  # (batch, hidden_size * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
               or (batch, 1, seq_len, input_size) for CNN features
        """
        # Ensure input has shape (batch, seq_len, input_size)
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove channel dimension
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Ensure float type
        x = x.float()
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Apply attention
        attn_out = self.attention_net(lstm_out)  # (batch, hidden_size * 2)
        
        # Final classification
        output = self.fc(attn_out)  # (batch, num_classes)
        
        return output

def get_model(
    model_type: str,
    num_classes: int,
    config: Optional[dict] = None
) -> nn.Module:
    """
    Factory Funktion für Modelle
    
    Args:
        model_type: "cnn" oder "lstm"
        num_classes: Anzahl der Klassen
        config: Modell Konfiguration
        
    Returns:
        PyTorch Modell
    """
    if config is None:
        config = {}
    
    if model_type.lower() == "cnn":
        return CNNModel(
            channels=config.get('channels', [32, 64, 128]),
            kernel_size=config.get('kernel_size', 3),
            stride=config.get('stride', 1),
            padding=config.get('padding', 1),
            pool_size=config.get('pool_size', 2),
            dropout=config.get('dropout', 0.5),
            dense_units=config.get('dense_units', 256),
            num_classes=num_classes
        )
    elif model_type.lower() == "lstm":
        return LSTMModel(
            input_size=config.get('input_size', 40),
            hidden_size=config.get('hidden_size', 32),
            num_layers=config.get('num_layers', 1),
            dropout=config.get('dropout', 0.5),
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}")
