"""
R@1 (Retrieval at Rank 1) Evaluation Script for Multimodal Alignment
=======================================================================

This script evaluates the multimodal alignment capability of trained models
using retrieval metrics including:
- R@1, R@5, R@10: Recall at different ranks
- MRR: Mean Reciprocal Rank
- mAP: Mean Average Precision

Usage:
    # For predefined datasets:
    python evaluate_retrieval.py --model_path best_val_model.pt --config config.json --split test

    # For custom datasets (user_data):
    python evaluate_retrieval.py --model_path best_val_model.pt --split test \
        --root_dir ../dataset/ --dataset_name jarvis --property_name mbj_bandgap --visualize
"""

import os
import sys
import argparse
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import project modules
from config import TrainingConfig
from data import get_train_val_loaders
from models.alignn import ALIGNN

# Import for dataset loading
from jarvis.core.atoms import Atoms
from tokenizers.normalizers import BertNormalizer


def load_dataset(cif_dir, id_prop_file, dataset, property_name):
    """Load local dataset from CIF files and CSV description

    Args:
        cif_dir: CIF file directory
        id_prop_file: Description file path (description.csv)
        dataset: Dataset name
        property_name: Property name

    Returns:
        dataset_array: List of sample dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset} - {property_name}")
    print(f"CIF directory: {cif_dir}")
    print(f"Description file: {id_prop_file}")
    print(f"{'='*60}\n")

    # Read CSV file
    with open(id_prop_file, 'r') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"Total samples: {len(data)}")

    # Text normalizer
    norm = BertNormalizer(lowercase=False, strip_accents=True,
                         clean_text=True, handle_chinese_chars=True)

    # Load vocabulary mappings - smart path finding
    possible_paths = [
        'vocab_mappings.txt',
        './vocab_mappings.txt',
        os.path.join(os.path.dirname(__file__), 'vocab_mappings.txt'),
    ]

    mappings = {}
    vocab_file = None
    for path in possible_paths:
        if os.path.exists(path):
            vocab_file = path
            break

    if vocab_file:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',', 1)
                if len(parts) == 2:
                    mappings[parts[0].strip()] = parts[1].strip()
        print(f"Loaded {len(mappings)} vocabulary mappings from {vocab_file}")
    else:
        print("⚠️  Warning: vocab_mappings.txt not found, using default normalization")

    def normalize(text):
        """Normalize text using BERT normalizer and vocab mappings"""
        if text is None:
            return None
        normalized = norm.normalize_str(text)
        # Apply vocabulary mappings
        out = []
        for s in normalized.split('\n'):
            norm_s = ''
            for c in s:
                norm_s += mappings.get(c, ' ')
            out.append(norm_s)
        return '\n'.join(out)

    # Build dataset
    dataset_array = []
    skipped = 0

    for j in tqdm(range(len(data)), desc="Loading data"):
        try:
            # Parse CSV row (based on dataset type)
            if dataset.lower() == 'jarvis':
                # JARVIS format: id, composition, target, description, file_name
                id = data[j][0]
                composition = data[j][1]
                target = data[j][2]
                crys_desc_full = data[j][3] if len(data[j]) > 3 else None
            elif dataset.lower() == 'mp':
                # Materials Project format
                id = data[j][0]
                composition = data[j][1]
                target = data[j][2]
                crys_desc_full = data[j][3] if len(data[j]) > 3 else None
            else:
                # Generic format: id, target, description
                id = data[j][0]
                target = data[j][1]
                crys_desc_full = data[j][2] if len(data[j]) > 2 else None

            # Normalize description text
            if crys_desc_full:
                crys_desc_full = normalize(crys_desc_full)

            # Load CIF file
            cif_file = os.path.join(cif_dir, f'{id}.cif')
            if not os.path.exists(cif_file):
                raise FileNotFoundError(f"CIF file not found: {cif_file}")

            atoms = Atoms.from_cif(cif_file)

            # If CSV doesn't provide description, generate from CIF
            if crys_desc_full is None:
                crys_desc_full = normalize(atoms.composition.reduced_formula)

            info = {
                "atoms": atoms.to_dict(),
                "jid": id,
                "text": crys_desc_full,
                "target": float(target)
            }

            dataset_array.append(info)

        except Exception as e:
            skipped += 1
            if skipped <= 5:  # Only show first 5 errors
                print(f"Skipping sample {id if 'id' in locals() else j}: {e}")

    print(f"\nSuccessfully loaded: {len(dataset_array)} samples")
    print(f"Skipped: {skipped} samples\n")

    return dataset_array


class RetrievalEvaluator:
    """Evaluator for multimodal retrieval tasks."""

    def __init__(self, model, device='cuda'):
        """Initialize evaluator.

        Args:
            model: Trained multimodal model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, data_loader, modality='both'):
        """Extract features from data loader.

        Args:
            data_loader: PyTorch DataLoader
            modality: 'graph', 'text', or 'both'

        Returns:
            Dictionary containing extracted features and metadata
        """
        graph_features = []
        text_features = []
        ids = []
        targets = []

        print(f"\n🔍 Extracting {modality} features from {len(data_loader.dataset)} samples...")

        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting features")):
            g, lg, text, target = batch
            g = g.to(self.device)
            lg = lg.to(self.device)

            # Get batch size
            batch_size = len(target)

            # Run forward pass with return_features=True to get separate features
            output = self.model([g, lg, text], return_features=True)

            if isinstance(output, dict):
                if 'graph_features' in output and modality in ['graph', 'both']:
                    graph_features.append(output['graph_features'].cpu())
                if 'text_features' in output and modality in ['text', 'both']:
                    text_features.append(output['text_features'].cpu())
            else:
                print("❌ Error: Model does not return dict with separate features.")
                print("   Make sure you're using return_features=True in the forward pass.")
                raise ValueError("Model output format not compatible with retrieval evaluation")

            # Store metadata
            targets.extend(target.cpu().numpy().tolist())

            # Get sample IDs
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + batch_size
            batch_ids = data_loader.dataset.ids[start_idx:end_idx]
            ids.extend(batch_ids)

        result = {'ids': ids, 'targets': targets}

        if graph_features:
            result['graph_features'] = torch.cat(graph_features, dim=0)
            print(f"✅ Graph features shape: {result['graph_features'].shape}")

        if text_features:
            result['text_features'] = torch.cat(text_features, dim=0)
            print(f"✅ Text features shape: {result['text_features'].shape}")

        return result

    def compute_similarity_matrix(self, graph_features, text_features, normalize=True):
        """Compute similarity matrix between graph and text features.

        Args:
            graph_features: Graph feature tensor [N, D]
            text_features: Text feature tensor [N, D]
            normalize: Whether to L2-normalize features (for cosine similarity)

        Returns:
            Similarity matrix [N, N] where sim[i,j] = similarity(graph_i, text_j)
        """
        if normalize:
            graph_features = F.normalize(graph_features, dim=1)
            text_features = F.normalize(text_features, dim=1)

        # Compute cosine similarity
        similarity = torch.matmul(graph_features, text_features.T)

        return similarity

    def compute_retrieval_metrics(self, similarity_matrix, top_k_list=[1, 5, 10]):
        """Compute retrieval metrics from similarity matrix.

        Args:
            similarity_matrix: Similarity matrix [N, N]
            top_k_list: List of k values for R@k computation

        Returns:
            Dictionary of metrics
        """
        N = similarity_matrix.size(0)

        # Ground truth: diagonal elements are positive pairs
        # i.e., graph_i should match text_i

        metrics = {}

        # ===== Graph-to-Text Retrieval =====
        print("\n📊 Computing Graph-to-Text retrieval metrics...")

        # For each graph, rank all texts by similarity
        g2t_ranks = []
        for i in range(N):
            # Get similarities for graph_i with all texts
            similarities = similarity_matrix[i]  # [N]

            # Sort indices by similarity (descending)
            sorted_indices = torch.argsort(similarities, descending=True)

            # Find rank of correct text (index i)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            g2t_ranks.append(rank)

        g2t_ranks = np.array(g2t_ranks)

        # Compute R@k for graph-to-text
        for k in top_k_list:
            recall_at_k = np.mean(g2t_ranks <= k) * 100
            metrics[f'G2T_R@{k}'] = recall_at_k
            print(f"  Graph→Text R@{k}: {recall_at_k:.2f}%")

        # Compute MRR for graph-to-text
        mrr_g2t = np.mean(1.0 / g2t_ranks) * 100
        metrics['G2T_MRR'] = mrr_g2t
        print(f"  Graph→Text MRR: {mrr_g2t:.2f}%")

        # ===== Text-to-Graph Retrieval =====
        print("\n📊 Computing Text-to-Graph retrieval metrics...")

        # For each text, rank all graphs by similarity
        t2g_ranks = []
        for j in range(N):
            # Get similarities for text_j with all graphs
            similarities = similarity_matrix[:, j]  # [N]

            # Sort indices by similarity (descending)
            sorted_indices = torch.argsort(similarities, descending=True)

            # Find rank of correct graph (index j)
            rank = (sorted_indices == j).nonzero(as_tuple=True)[0].item() + 1
            t2g_ranks.append(rank)

        t2g_ranks = np.array(t2g_ranks)

        # Compute R@k for text-to-graph
        for k in top_k_list:
            recall_at_k = np.mean(t2g_ranks <= k) * 100
            metrics[f'T2G_R@{k}'] = recall_at_k
            print(f"  Text→Graph R@{k}: {recall_at_k:.2f}%")

        # Compute MRR for text-to-graph
        mrr_t2g = np.mean(1.0 / t2g_ranks) * 100
        metrics['T2G_MRR'] = mrr_t2g
        print(f"  Text→Graph MRR: {mrr_t2g:.2f}%")

        # ===== Average Metrics =====
        print("\n📊 Average Retrieval Metrics:")
        for k in top_k_list:
            avg_recall = (metrics[f'G2T_R@{k}'] + metrics[f'T2G_R@{k}']) / 2
            metrics[f'Avg_R@{k}'] = avg_recall
            print(f"  Average R@{k}: {avg_recall:.2f}%")

        avg_mrr = (mrr_g2t + mrr_t2g) / 2
        metrics['Avg_MRR'] = avg_mrr
        print(f"  Average MRR: {avg_mrr:.2f}%")

        # Store rank distributions for analysis
        metrics['g2t_ranks'] = g2t_ranks
        metrics['t2g_ranks'] = t2g_ranks

        return metrics

    def visualize_similarity_matrix(self, similarity_matrix, save_path,
                                   sample_size=50, title="Similarity Matrix"):
        """Visualize similarity matrix as heatmap.

        Args:
            similarity_matrix: Similarity matrix to visualize
            save_path: Path to save visualization
            sample_size: Number of samples to visualize (for readability)
            title: Plot title
        """
        # Sample for visualization if too large
        N = similarity_matrix.size(0)
        if N > sample_size:
            indices = np.random.choice(N, sample_size, replace=False)
            indices = np.sort(indices)
            sim_viz = similarity_matrix[indices][:, indices].cpu().numpy()
        else:
            sim_viz = similarity_matrix.cpu().numpy()

        # Create figure
        plt.figure(figsize=(12, 10))

        # Plot heatmap
        sns.heatmap(sim_viz, cmap='RdYlBu_r', center=0,
                   xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Cosine Similarity'})

        plt.xlabel('Text Index', fontsize=14)
        plt.ylabel('Graph Index', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Similarity matrix visualization saved to: {save_path}")
        plt.close()

    def visualize_rank_distribution(self, ranks, save_path,
                                    title="Rank Distribution", max_rank=50):
        """Visualize distribution of retrieval ranks.

        Args:
            ranks: Array of ranks
            save_path: Path to save visualization
            title: Plot title
            max_rank: Maximum rank to show in histogram
        """
        plt.figure(figsize=(12, 6))

        # Clip ranks to max_rank for visualization
        ranks_clipped = np.clip(ranks, 1, max_rank)

        # Plot histogram
        plt.hist(ranks_clipped, bins=range(1, max_rank+2),
                edgecolor='black', alpha=0.7, color='steelblue')

        plt.xlabel('Rank', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        # Add statistics text
        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        r_at_1 = np.mean(ranks == 1) * 100

        stats_text = f'Mean Rank: {mean_rank:.2f}\n'
        stats_text += f'Median Rank: {median_rank:.0f}\n'
        stats_text += f'R@1: {r_at_1:.2f}%'

        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Rank distribution visualization saved to: {save_path}")
        plt.close()

    def save_retrieval_results(self, similarity_matrix, ids, targets, save_dir, top_k=10):
        """Save detailed retrieval results for qualitative analysis.

        Args:
            similarity_matrix: Similarity matrix [N, N]
            ids: List of sample IDs
            targets: List of target values
            save_dir: Directory to save results
            top_k: Number of top retrievals to save
        """
        N = similarity_matrix.size(0)

        # Graph-to-Text retrieval results
        g2t_results = []
        for i in range(N):
            similarities = similarity_matrix[i].cpu().numpy()
            sorted_indices = np.argsort(similarities)[::-1]  # Descending order

            result = {
                'graph_id': ids[i],
                'graph_target': targets[i],
                'correct_rank': int(np.where(sorted_indices == i)[0][0] + 1),
                'top_k_ids': [ids[idx] for idx in sorted_indices[:top_k]],
                'top_k_targets': [targets[idx] for idx in sorted_indices[:top_k]],
                'top_k_similarities': similarities[sorted_indices[:top_k]].tolist(),
            }
            g2t_results.append(result)

        # Save to JSON
        g2t_path = os.path.join(save_dir, 'graph_to_text_retrieval.json')
        with open(g2t_path, 'w') as f:
            json.dump(g2t_results, f, indent=2)
        print(f"✅ Graph-to-Text retrieval results saved to: {g2t_path}")

        # Text-to-Graph retrieval results
        t2g_results = []
        for j in range(N):
            similarities = similarity_matrix[:, j].cpu().numpy()
            sorted_indices = np.argsort(similarities)[::-1]

            result = {
                'text_id': ids[j],
                'text_target': targets[j],
                'correct_rank': int(np.where(sorted_indices == j)[0][0] + 1),
                'top_k_ids': [ids[idx] for idx in sorted_indices[:top_k]],
                'top_k_targets': [targets[idx] for idx in sorted_indices[:top_k]],
                'top_k_similarities': similarities[sorted_indices[:top_k]].tolist(),
            }
            t2g_results.append(result)

        # Save to JSON
        t2g_path = os.path.join(save_dir, 'text_to_graph_retrieval.json')
        with open(t2g_path, 'w') as f:
            json.dump(t2g_results, f, indent=2)
        print(f"✅ Text-to-Graph retrieval results saved to: {t2g_path}")


def load_model_and_config(model_path, config_path=None, device='cuda'):
    """Load trained model and configuration.

    Args:
        model_path: Path to saved model checkpoint
        config_path: Path to config file (optional, will use model_path dir if not provided)
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    print(f"\n🔄 Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load config
    if config_path is None:
        # Try to find config in same directory as model
        model_dir = os.path.dirname(model_path)
        config_path = os.path.join(model_dir, 'config.json')

    if os.path.exists(config_path):
        print(f"📄 Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        print("⚠️  Config file not found, using default config")
        config = TrainingConfig()

    # Check if model config is saved in checkpoint
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        print("✅ Using model config from checkpoint")
    else:
        model_config = config.model
        print("⚠️  Model config not in checkpoint, using training config")

    # Initialize model
    model = ALIGNN(model_config)

    # Load state dict
    model.load_state_dict(checkpoint['model'])

    print("✅ Model loaded successfully!")

    return model, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate multimodal retrieval performance')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model checkpoint (.pt file)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: same as model dir)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Top-k retrievals to save for analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')

    # Dataset loading arguments for custom datasets
    parser.add_argument('--root_dir', type=str, default='../dataset/',
                       help='Root directory for custom datasets (default: ../dataset/)')
    parser.add_argument('--dataset_name', type=str, default='jarvis',
                       help='Dataset name (jarvis/mp, default: jarvis)')
    parser.add_argument('--property_name', type=str, default='mbj_bandgap',
                       help='Property name (e.g., mbj_bandgap, default: mbj_bandgap)')

    args = parser.parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), 'retrieval_results')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📁 Output directory: {args.output_dir}")

    # Load model and config
    model, config = load_model_and_config(args.model_path, args.config_path, device)

    # Load dataset if using custom data (dataset="user_data")
    dataset_array = []
    if config.dataset == "user_data":
        print(f"\n📦 Detected custom dataset '{config.dataset}', loading from files...")

        # Construct paths
        if args.dataset_name.lower() == 'jarvis':
            cif_dir = os.path.join(args.root_dir, f'jarvis/{args.property_name}/cif/')
            id_prop_file = os.path.join(args.root_dir, f'jarvis/{args.property_name}/description.csv')
        elif args.dataset_name.lower() == 'mp':
            cif_dir = os.path.join(args.root_dir, 'mp_2018_new/')
            id_prop_file = os.path.join(args.root_dir, 'mp_2018_new/mat_text.csv')
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")

        # Check if paths exist
        if not os.path.exists(cif_dir):
            raise FileNotFoundError(
                f"CIF directory not found: {cif_dir}\n"
                f"Please check --root_dir and --dataset_name arguments.\n"
                f"Use --root_dir to specify the dataset root directory."
            )
        if not os.path.exists(id_prop_file):
            raise FileNotFoundError(
                f"Description file not found: {id_prop_file}\n"
                f"Please check --root_dir and --property_name arguments."
            )

        # Load dataset
        dataset_array = load_dataset(cif_dir, id_prop_file, args.dataset_name, args.property_name)

        if len(dataset_array) == 0:
            raise ValueError("Dataset is empty! Please check data files.")

        print(f"✓ Successfully loaded {len(dataset_array)} samples\n")

    # Prepare data loaders
    print(f"\n🔄 Loading {args.split} data...")
    config.batch_size = args.batch_size
    train_loader, val_loader, test_loader, _ = get_train_val_loaders(
        dataset=config.dataset,
        dataset_array=dataset_array,  # Pass loaded dataset
        target=config.target,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        line_graph=True,
        split_seed=config.random_seed,
        workers=config.num_workers,
        pin_memory=config.pin_memory,
        id_tag=config.id_tag,
        use_canonize=config.use_canonize,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
    )

    # Select data loader
    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}[args.split]

    # Initialize evaluator
    evaluator = RetrievalEvaluator(model, device=device)

    # Extract features
    features = evaluator.extract_features(data_loader, modality='both')

    if 'graph_features' not in features or 'text_features' not in features:
        print("\n❌ Error: Could not extract both graph and text features from model.")
        print("   Please ensure the model outputs separate features for each modality.")
        return

    # Compute similarity matrix
    print("\n🔄 Computing similarity matrix...")
    similarity_matrix = evaluator.compute_similarity_matrix(
        features['graph_features'],
        features['text_features'],
        normalize=True
    )
    print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")

    # Compute retrieval metrics
    metrics = evaluator.compute_retrieval_metrics(similarity_matrix, top_k_list=[1, 5, 10, 20])

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'retrieval_metrics_{args.split}.json')
    metrics_to_save = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                       for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_path}")

    # Save detailed retrieval results
    print("\n🔄 Saving detailed retrieval results...")
    evaluator.save_retrieval_results(
        similarity_matrix,
        features['ids'],
        features['targets'],
        args.output_dir,
        top_k=args.top_k
    )

    # Generate visualizations
    if args.visualize:
        print("\n🎨 Generating visualizations...")

        # Similarity matrix heatmap
        sim_path = os.path.join(args.output_dir, f'similarity_matrix_{args.split}.png')
        evaluator.visualize_similarity_matrix(
            similarity_matrix,
            sim_path,
            title=f'Graph-Text Similarity Matrix ({args.split.upper()} set)'
        )

        # Rank distributions
        g2t_rank_path = os.path.join(args.output_dir, f'g2t_rank_dist_{args.split}.png')
        evaluator.visualize_rank_distribution(
            metrics['g2t_ranks'],
            g2t_rank_path,
            title=f'Graph→Text Rank Distribution ({args.split.upper()} set)'
        )

        t2g_rank_path = os.path.join(args.output_dir, f't2g_rank_dist_{args.split}.png')
        evaluator.visualize_rank_distribution(
            metrics['t2g_ranks'],
            t2g_rank_path,
            title=f'Text→Graph Rank Distribution ({args.split.upper()} set)'
        )

    # Print final summary
    print("\n" + "="*80)
    print("🎯 RETRIEVAL EVALUATION SUMMARY")
    print("="*80)
    print(f"\nDataset: {config.dataset}")
    print(f"Split: {args.split.upper()}")
    print(f"Number of samples: {len(features['ids'])}")
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-"*32)
    for k in [1, 5, 10, 20]:
        if f'Avg_R@{k}' in metrics:
            print(f"R@{k:<18} {metrics[f'Avg_R@{k}']:>9.2f}%")
    print(f"{'MRR':<20} {metrics['Avg_MRR']:>9.2f}%")
    print("="*80)
    print(f"\n✅ All results saved to: {args.output_dir}\n")


if __name__ == '__main__':
    main()
