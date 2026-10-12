"""Original matched-4 checkpoint model, retaining projection.proj state keys."""
import torch
import torch.nn as nn
FUSED_DIM = 1344
HIDDEN = 128
BOTTLENECK = 4
NUM_CLASSES = 25

class FusionProjection(nn.Module):
    """Shared Spec B projection 1344 -> 128 -> 4 (identical across E2-matched / E3-4)."""

    def __init__(self, fused_dim=FUSED_DIM, hidden=HIDDEN, bottleneck=BOTTLENECK, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns the [B, 4] bottleneck (pre-activation)
        return self.proj(x)

class E2MatchedHead(nn.Module):
    """Fair classical comparator: projection -> tanh -> Linear(4,4,bias=False) -> tanh -> Linear(4,25)."""

    def __init__(self, bottleneck=BOTTLENECK, num_classes=NUM_CLASSES):
        super().__init__()
        self.projection = FusionProjection(bottleneck=bottleneck)
        self.middle = nn.Linear(bottleneck, bottleneck, bias=False)  # 16 params (matched)
        self.readout = nn.Linear(bottleneck, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(x)
        z = torch.tanh(z)
        z = torch.tanh(self.middle(z))
        return self.readout(z)

    def post_bottleneck_parameters(self):
        # the matched decision sub-network (excludes shared projection/readout)
        return self.middle.parameters()

def count_param_split(model: nn.Module):
    """Return (total_trainable_params, post_bottleneck_trainable_params).

    Required by the result JSON (see PARAM_MATCH_SPEC.md sec 3).
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    post = sum(p.numel() for p in model.post_bottleneck_parameters() if p.requires_grad)
    return total, post
