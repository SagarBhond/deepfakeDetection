import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import LSTM
import torch.nn.functional as F


class ResNextLSTMDeepfakeDetector(nn.Module):
    """
    Deepfake Detection Model combining ResNext-50 and LSTM
    Based on the research paper: "Deepfake Detection Using Deep Learning: ResNext and LSTM"
    """
    
    def __init__(self, num_classes=2, sequence_length=16, hidden_size=256, num_layers=2):
        super(ResNextLSTMDeepfakeDetector, self).__init__()
        
        # Load pre-trained ResNext-50
        self.resnext = models.resnext50_32x4d(pretrained=True)
        
        # Remove the final classification layer
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-1])
        
        # Freeze early layers for transfer learning
        for param in list(self.resnext.parameters())[:-10]:
            param.requires_grad = False
            
        # Feature dimension from ResNext-50
        self.feature_dim = 2048
        
        # LSTM for temporal analysis
        self.lstm = LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 512),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.sequence_length = sequence_length
        
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for batch processing through ResNext
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using ResNext
        with torch.no_grad():
            features = self.resnext(x_reshaped)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, self.feature_dim)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(last_output)
        
        return output
    
    def extract_features(self, x):
        """
        Extract features from input frames without classification
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, sequence_length, feature_dim)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for batch processing
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features
        with torch.no_grad():
            features = self.resnext(x_reshaped)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, self.feature_dim)
        
        return features


class ResNextFeatureExtractor(nn.Module):
    """
    Standalone ResNext feature extractor for preprocessing
    """
    
    def __init__(self):
        super(ResNextFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNext-50
        self.resnext = models.resnext50_32x4d(pretrained=True)
        
        # Remove the final classification layer
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-1])
        
        # Freeze all parameters
        for param in self.resnext.parameters():
            param.requires_grad = False
            
        self.resnext.eval()
        
    def forward(self, x):
        """
        Extract features from input images
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        with torch.no_grad():
            features = self.resnext(x)
            features = features.view(features.size(0), -1)
        
        return features


class LSTMTemporalClassifier(nn.Module):
    """
    LSTM-based temporal classifier for deepfake detection
    """
    
    def __init__(self, input_size=2048, hidden_size=256, num_layers=2, num_classes=2):
        super(LSTMTemporalClassifier, self).__init__()
        
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through LSTM classifier
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(last_output)
        
        return output


def create_model(sequence_length=16, num_classes=2, device='cuda'):
    """
    Create and initialize the ResNext-LSTM model
    
    Args:
        sequence_length: Number of frames in sequence
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    model = ResNextLSTMDeepfakeDetector(
        num_classes=num_classes,
        sequence_length=sequence_length,
        hidden_size=256,
        num_layers=2
    )
    
    model = model.to(device)
    
    return model


def load_pretrained_model(model_path, device='cuda'):
    """
    Load a pre-trained model from file
    
    Args:
        model_path: Path to the model file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = ResNextLSTMDeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(sequence_length=16, num_classes=2, device=device)
    
    # Test input
    batch_size = 2
    sequence_length = 16
    channels = 3
    height = 224
    width = 224
    
    test_input = torch.randn(batch_size, sequence_length, channels, height, width).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
