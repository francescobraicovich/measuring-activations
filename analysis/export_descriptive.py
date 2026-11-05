"""Export middle-layer descriptive figures from the saved calibration tables."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plotly.io as pio
from src.plots import descriptive_figures

os.environ.setdefault('BROWSER_PATH', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
study = ROOT / 'results/study'
names = ('signed_distributions', 'magnitude_distributions', 'channel_rms', 'scale_weight',
         'weighted_scale', 'jaccard', 'rank_correlation', 'density_N_L', 'density_L_W',
         'removal', 'token_removal')
for label in ('gpt2_6', 'smollm2_16'):
    figures = descriptive_figures(study / 'descriptive' / label)
    destination = study / 'figures/descriptive' / label
    for name in names:
        figure = figures[name]
        figure.update_layout(width=1200, height=650, margin=dict(l=90, r=80, t=90, b=90))
        for extension in ('svg', 'png'):
            try:
                figure.write_image(destination / f'{name}.{extension}', scale=2 if extension == 'png' else 1)
            except Exception as error:
                print(f'FAILED {label}/{name}.{extension}: {error}', flush=True)
            else:
                print(f'Wrote {label}/{name}.{extension}', flush=True)
