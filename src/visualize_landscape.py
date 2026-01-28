import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.configurator import Config
from utils.utils import init_seed, init_logger, get_model
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader

try:
    import loss_landscapes
    import loss_landscapes.metrics
except ImportError:
    print("Error: loss-landscapes is not installed. Please install it using:")
    print("pip install loss-landscapes")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Visualize Loss Landscape')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. BM3, MCDRec)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved checkpoint')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for loss calculation')
    parser.add_argument('--steps', type=int, default=40, help='Resolution steps')
    parser.add_argument('--distance', type=float, default=10, help='Distance to traverse')
    parser.add_argument('--normalize', type=str, default='filter', choices=['filter', 'layer', 'none'], help='Normalization type')
    
    args = parser.parse_args()

    # 1. Config
    config_dict = {
        'model': args.model,
        'dataset': args.dataset,
        'train_batch_size': args.batch_size,
        'gpu_id': 0,
        'use_gpu': True,
        'state': 'INFO'
    }
    config = Config(args.model, args.dataset, config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    # 2. Data
    print("Loading dataset...")
    dataset = RecDataset(config)
    _, _, test_dataset = dataset.split()
    
    # Use TrainDataLoader to get negative sampling for BPR loss
    # Ensure neg_sampling is configured
    if config['neg_sampling'] is None:
        # Force some negative sampling if not present, though config usually has it
        pass 
        
    # IMPORTANT: Shuffle=False and fix seed to ensure the SAME batch is selected across different runs
    # This allows fair comparison between different models (e.g. Origin vs MG)
    test_data_loader = TrainDataLoader(config, test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get Fixed Batch
    try:
        fixed_batch = next(iter(test_data_loader))
    except StopIteration:
        print("Test dataset is empty!")
        return
        
    fixed_batch = fixed_batch.to(config['device'])
    print(f"Fixed batch shape: {len(fixed_batch)} interactions")

    # 3. Model
    print("Loading model...")
    model = get_model(config['model'])(config, test_data_loader).to(config['device'])
    
    # Load Checkpoint
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=config['device'])
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 4. Metric Fn
    def metric_fn(model_wrapper):
        with torch.no_grad():
            loss = model_wrapper.calculate_loss(fixed_batch)
            if isinstance(loss, tuple):
                loss = loss[0]
            return loss.item()

    # 5. Landscape
    print(f"Generating landscape with distance={args.distance}, steps={args.steps}, norm={args.normalize}...")
    loss_data = loss_landscapes.random_plane(
        model,
        metric_fn,
        distance=args.distance,
        steps=args.steps,
        normalization=args.normalize,
        deepcopy_model=True
    )

    # 6. Plot
    print("Plotting...")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[i for i in range(args.steps)] for _ in range(args.steps)])
    Y = np.array([[j for _ in range(args.steps)] for j in range(args.steps)])
    
    surf = ax.plot_surface(X, Y, loss_data, cmap='viridis', edgecolor='none')
    ax.set_title(f'Loss Landscape: {args.model}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    save_path = f'landscape_{args.model}_{args.dataset}.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    main()
