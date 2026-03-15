import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, AutoModel

class WeightedLayerAggregator(nn.Module):
    """
    Learnable weighted aggregation of all transformer layers.
    
    Aggregates features from all 24 transformer layers plus input embeddings
    using learned weights, which helps capture both low-level acoustic features
    (early layers) and high-level semantic information (late layers).
    """

    def __init__(self, num_layers: int = 25, hidden_size: int = 1024):
        """
        Args:
            num_layers  : Number of layers to aggregate (24 transformer + 1 input embedding).
            hidden_size : Hidden dimension size (1024 for MERT-330M).
        """
        super().__init__()

        self.num_layers = num_layers 
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states : Tuple of tensors, each [batch, seq_len, hidden_size]
        
        Returns:
            torch.Tensor : Aggregated features [batch, seq_len, hidden_size]
        """

        # stack all layers: [num_layers, batch, seq_len, hidden_size]
        stacked = torch.stack(all_hidden_states, dim=0)
        weights = F.softmax(self.layer_weights, dim=0)

        # Weighted sum: [num_layers, 1, 1, 1] * [num_layers, batch, seq, hidden]
        weights = weights.view(-1, 1, 1, 1)
        aggregated = (stacked * weights).sum(dim=0)
        
        return aggregated
    


class MERTGenreClassifier(nn.Module):
    """
    MERTGenreClassifier
    
    A fine-tuned Music undERstanding model with large-scale self-supervised Training 
    (MERT) for music genre classification.
    
    Backbone : m-a-p/MERT-v1-330M
    Input    : Raw audio waveform [B, audio_length] at 24kHz
               Recommended audio_length: 30 seconds = 720,000 samples
    Output   : Logits [B, num_classes]
    
    Architecture:
        - MERT-v1-330M backbone (330M params, 24 transformer layers)
        - Weighted layer aggregation (learnable combination of all 24 layers)
        - Temporal average pooling
        - Custom classification head:
            LayerNorm(1024)
            Linear(1024 -> 512) -> ReLU -> Dropout(0.3)
            Linear(512 -> 256) -> ReLU -> Dropout(0.3)
            Linear(256 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "m-a-p/MERT-v1-330M",
        sample_rate: int = 24000,
        chunk_duration: int = 30,
        use_weighted_layers: bool = True,
    ):
        
        """
        Args:
            num_classes         : Number of output genre classes.
            model_name          : HuggingFace model identifier (default MERT-v1-330M).
            sample_rate         : Audio sample rate in Hz (must be 24000 for MERT).
            chunk_duration      : Expected audio chunk duration in seconds (default 30).
            use_weighted_layers : Whether to use weighted layer aggregation (recommended).
        """

        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.use_weighted_layers = use_weighted_layers

        self.expected_length = int(chunk_duration * sample_rate) 

        # Load MERT backbone
        self.mert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )


        # Get model dimensions
        self.hidden_size = self.mert.config.hidden_size  # 1024 for MERT-330M
        self.num_layers = self.mert.config.num_hidden_layers 


        # Weighted layer aggregation
        if use_weighted_layers:
            self.layer_aggregator = WeightedLayerAggregator(
                num_layers=self.num_layers + 1,  # +1 for input embeddings
                hidden_size=self.hidden_size
            )
        else:
            self.layer_aggregator = None

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Feature extractor (processor)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_values (torch.Tensor):
                Raw audio waveform tensor.
                Shape: 
                    [batch_size, audio_length]
                where audio_length = chunk_duration * sample_rate
                Example: [8, 720000] for 30-second chunks at 24kHz
        
        Returns:
            torch.Tensor:
                Logits of shape [batch_size, num_classes]
        """
        # Get MERT features with all hidden states
        outputs = self.mert(
            input_values,
            output_hidden_states=True,
            return_dict=True
        )

        # Aggregate layers or use last layer
        if self.use_weighted_layers and self.layer_aggregator is not None:
            # Weighted combination of all 25 layers (input + 24 transformer layers)
            hidden_states = self.layer_aggregator(outputs.hidden_states)
        else:
            # Use only the last transformer layer
            hidden_states = outputs.last_hidden_state
        
        # hidden_states shape: [batch, seq_len, hidden_size]
        # seq_len ≈ 75 embeddings/second * 30 seconds = 2250 time steps
        
        # Temporal average pooling over time dimension
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
        
        # Classification
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        return logits
    
    
    def get_features(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Extract MERT features without classification.
        Useful for feature extraction or transfer learning.
        
        Args:
            input_values : Raw audio [batch, audio_length]
        
        Returns:
            torch.Tensor : Pooled features [batch, hidden_size]
        """
        outputs = self.mert(
            input_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        if self.use_weighted_layers and self.layer_aggregator is not None:
            hidden_states = self.layer_aggregator(outputs.hidden_states)
        else:
            hidden_states = outputs.last_hidden_state
        
        pooled = hidden_states.mean(dim=1)
        
        return pooled
    
    
    def count_parameters(self) -> dict:
        """
        Count trainable and total parameters.
        
        Returns:
            dict : Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_pct': 100 * trainable / total if total > 0 else 0
        }
