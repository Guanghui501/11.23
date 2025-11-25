"""
Weight monitoring utility for DynamicFusionModule.

Usage in training loop:
    # After each epoch or every N iterations
    from monitor_fusion_weights import print_fusion_weights
    print_fusion_weights(model)

This will show the average w_graph and w_text values learned by the router,
helping you verify that the model learns reasonable modal competition.

Expected behavior for material property prediction:
- w_graph should typically be > 0.5 (graph dominant)
- w_text should be < 0.5 (text supplementary)
- Effective graph weight = 1 + w_graph (due to double residual)
- Final ratio: graph:[1-2], text:[0-1]
"""

import torch


def print_fusion_weights(model, verbose=True):
    """Print weight statistics from all DynamicFusionModule instances in model.

    Args:
        model: The ALIGNN model instance
        verbose: If True, print detailed interpretation

    Returns:
        dict: Weight statistics from all fusion modules
    """
    stats = {}

    # Find all DynamicFusionModule instances
    if hasattr(model, 'middle_fusion_modules'):
        for name, module in model.middle_fusion_modules.items():
            module_stats = module.get_weight_stats()
            stats[name] = module_stats

            if verbose:
                w_g = module_stats['avg_w_graph']
                w_t = module_stats['avg_w_text']
                updates = module_stats['update_count']

                print(f"\n{'='*60}")
                print(f"Fusion Module: {name}")
                print(f"{'='*60}")
                print(f"Updates: {updates}")
                print(f"\nRouter learned weights (from Softmax competition):")
                print(f"  w_graph: {w_g:.4f}")
                print(f"  w_text:  {w_t:.4f}")
                print(f"  Sum:     {w_g + w_t:.4f} (should be ~1.0)")

                # Calculate effective weights (with double residual)
                eff_w_g = 1.0 + w_g
                eff_w_t = w_t
                total = eff_w_g + eff_w_t

                print(f"\nEffective weights (with double residual):")
                print(f"  Graph:  {eff_w_g:.4f} ({eff_w_g/total*100:.1f}%)")
                print(f"  Text:   {eff_w_t:.4f} ({eff_w_t/total*100:.1f}%)")

                # Interpretation
                print(f"\nInterpretation:")
                if eff_w_g / eff_w_t > 2.0:
                    print(f"  ✅ Graph dominant (ratio: {eff_w_g/eff_w_t:.2f}x)")
                    print(f"     This is expected for material property prediction.")
                elif eff_w_g / eff_w_t > 1.2:
                    print(f"  ✓ Graph preferred (ratio: {eff_w_g/eff_w_t:.2f}x)")
                    print(f"     Text has moderate influence.")
                elif eff_w_g / eff_w_t > 0.8:
                    print(f"  ⚠️ Balanced fusion (ratio: {eff_w_g/eff_w_t:.2f}x)")
                    print(f"     Warning: Text may have too much influence for physics tasks.")
                else:
                    print(f"  ❌ Text dominant (ratio: {eff_w_g/eff_w_t:.2f}x)")
                    print(f"     ERROR: This violates physics priors!")

                # Recommendations
                if w_t > 0.5:
                    print(f"\n  💡 Tip: Router is giving text high weight (w_text={w_t:.3f}).")
                    print(f"     Consider: 1) Check if text descriptions are too informative")
                    print(f"               2) Add weight clipping: w_text < 0.3")
                    print(f"               3) Increase router regularization")

    else:
        print("No middle fusion modules found in model.")

    return stats


def log_fusion_weights_to_file(model, filepath, epoch):
    """Append weight statistics to a CSV file for tracking over epochs.

    Args:
        model: The ALIGNN model instance
        filepath: Path to CSV file
        epoch: Current epoch number
    """
    import csv
    import os

    stats = {}
    if hasattr(model, 'middle_fusion_modules'):
        for name, module in model.middle_fusion_modules.items():
            module_stats = module.get_weight_stats()
            stats[name] = module_stats

    # Create file with header if it doesn't exist
    file_exists = os.path.exists(filepath)

    with open(filepath, 'a', newline='') as f:
        if not file_exists:
            # Write header
            fieldnames = ['epoch']
            for name in stats.keys():
                fieldnames.extend([
                    f'{name}_w_graph',
                    f'{name}_w_text',
                    f'{name}_eff_ratio'
                ])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        # Write data
        row = {'epoch': epoch}
        for name, module_stats in stats.items():
            w_g = module_stats['avg_w_graph']
            w_t = module_stats['avg_w_text']
            eff_ratio = (1.0 + w_g) / w_t if w_t > 1e-6 else float('inf')

            row[f'{name}_w_graph'] = f"{w_g:.6f}"
            row[f'{name}_w_text'] = f"{w_t:.6f}"
            row[f'{name}_eff_ratio'] = f"{eff_ratio:.4f}"

        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


# Example usage
if __name__ == "__main__":
    print("""
    Weight Monitoring Example
    ========================

    In your training script, add:

    ```python
    from monitor_fusion_weights import print_fusion_weights, log_fusion_weights_to_file

    # After each epoch
    for epoch in range(num_epochs):
        train(...)
        validate(...)

        # Print weight statistics
        print_fusion_weights(model)

        # Log to CSV for plotting
        log_fusion_weights_to_file(model, 'fusion_weights.csv', epoch)
    ```

    Then analyze with:
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('fusion_weights.csv')
    df.plot(x='epoch', y=['layer_2_w_graph', 'layer_2_w_text'])
    plt.ylabel('Weight')
    plt.title('Router Weight Evolution')
    plt.show()
    ```
    """)
