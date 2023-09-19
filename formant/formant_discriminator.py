import torch.nn as nn

from ._combine_discriminator import CombineDiscriminator
from .formant_discriminator import FormantTrackerWrapper


class FormantDiscriminator(nn.Module):
    """FormantDiscriminator uses FormantTracker to track the formant of a spectrogram,
    then feed the spectrogram and formant heatmap to a discriminator"""

    def __init__(self, device="cuda"):
        super(FormantDiscriminator, self).__init__()
        self.combine_discriminator = CombineDiscriminator().to(device).eval()
        self.formant_tracker = FormantTrackerWrapper().to(device).eval()
        for p in self.formant_tracker.parameters():  # freeze formant_tracker
            p.requires_grad = False

    def forward(self, spec, spec4formant, device="cuda"):
        spec = spec.to(device)
        formant_heatmap = self.formant_tracker.forward(spec4formant, device=device)
        return self.combine_discriminator.forward(spec, formant_heatmap)


if __name__ == "__main__":
    fd = FormantDiscriminator()
