from pathlib import Path
from plot_data_utils import make_single_seed, make_multiseed, make_ablation

root = Path(__file__).resolve().parents[1]
output = Path(__file__).resolve().parent / 'generated'
output.mkdir(exist_ok=True)
for dataset in ['cifar10', 'plantvillage']:
    make_single_seed(root, dataset, output / f'seed1234_{dataset}_accuracy.pdf')
    make_multiseed(root, dataset, output / f'multiseed_{dataset}.pdf')
make_ablation(root, output / 'ablation_current.pdf')
print(f'Figures written to {output}')
