from argparse import Namespace

import torch
import torch.nn as nn

from .model import FormantTracker

FormantTrackerConfig = Namespace(
    ckpt="./saved_ckpts/196_1125.ckpt",
    test_dir="./test_dir",
    predictions_dir="./predictions",
    sample_rate=16000,
    num_workers=4,
    normalize=True,
    n_fft=512,
    emph=0.97,
    write_predictions=True,
    gaussian_kernel_size=7,
    gaussian_kernel_sigma=1,
    is_cuda=True,
    test_batch_size=4,
    f1_blocks=2,
    f2_blocks=2,
    f3_blocks=2,
    f4_blocks=2,
    f4=False,
    bias1d=False,
    bias=True,
    dropout=0.2,
)

# n_bins = (FormantTrackerConfig.n_fft // 2) + 1
# bin_resolution = (FormantTrackerConfig.sample_rate / 2) / n_bins


class FormantTrackerWrapper(nn.Module):
    """Wraps a pretrained FormantTracker"""

    def __init__(self, device="cuda"):
        super(FormantTrackerWrapper, self).__init__()
        self.model = FormantTracker(FormantTrackerConfig).to(device).eval()
        self.model.load_state_dict(torch.load(FormantTrackerConfig.ckpt))

    def forward(self, spect4formant, device="cuda"):
        out, _ = self.model(spect4formant)
        # predictions=get_predicted_formants(out)

        return out


if __name__ == "__main__":
    ftw = FormantTrackerWrapper()
