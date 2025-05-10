import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models import resnet18
from contextlib import nullcontext

class TemporalActionDetector(nn.Module):
    def __init__(self, num_classes=6, window_size=32, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        
        video_backbone_pretrained = mvit_v2_s(
            weights=MViT_V2_S_Weights.KINETICS400_V1
        )
        self.rgb_feature_dim = 768
        if hasattr(video_backbone_pretrained, 'head') and isinstance(video_backbone_pretrained.head, nn.Sequential):
            video_backbone_pretrained.head = nn.Identity()
        self.video_backbone = self._adapt_mvit_for_32_frames(video_backbone_pretrained)

        self.tcn = nn.Sequential(
            nn.Conv1d(self.rgb_feature_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)

        self.action_classifier = nn.Conv1d(1024, num_classes, kernel_size=1)
    def _adapt_mvit_for_32_frames(self, model):
        model.pos_encoding.temporal_size = self.window_size // 2
        return model
    def reset_hidden_state(self, batch_size=1, device=None):
        self.hidden_state = None
    def forward(self, frames, hidden_state=None):
        batch_size = frames.shape[0]
        
        if hidden_state is not None:
            self.hidden_state = hidden_state
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            global_features = self.video_backbone(frames)
            
            T = self.window_size // 2
            global_features_expanded = global_features.unsqueeze(2).expand(-1, -1, T)
        
        temporal_features = self.tcn(global_features_expanded)
        temporal_features = temporal_features.transpose(1, 2)
        
        if self.hidden_state is None:
            rnn_output, new_hidden = self.rnn(temporal_features)
        else:
            rnn_output, new_hidden = self.rnn(temporal_features, self.hidden_state)
        
        self.hidden_state = new_hidden
        
        action_scores = self.action_classifier(rnn_output)
        
        return {
            'action_scores': action_scores,
            'hidden_state': new_hidden
        }
    
    def predict_sequence(self, video_clips, reset_state=True):
            if reset_state:
                self.reset_hidden_state(batch_size=1, device=video_clips[0].device)
            
            all_predictions = []
            
            for clip in video_clips:
                with torch.no_grad():
                    outputs = self.forward(clip)
                    action_scores = outputs['action_scores']
                    all_predictions.append(action_scores)
            
            return torch.cat(all_predictions, dim=1)  # [1, total_frames, num_classes]