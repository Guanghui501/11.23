#!/usr/bin/env python3
"""
使用 Optuna 进行超参数调优的训练脚本

运行方式:
    python train_optuna.py --n_trials 50 --output_dir optuna_results
"""

import argparse
import os
import json
import torch
import optuna
from optuna.trial import TrialState
import numpy as np
from pathlib import Path
from datetime import datetime

from config import TrainingConfig
from models.alignn import ALIGNNConfig
from data import get_train_val_loaders
from models.alignn import ALIGNN
from torch import nn
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.handlers import EarlyStopping


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def create_objective(base_config_path=None, n_epochs=100, dataset="user_data", target="target"):
    """创建 Optuna 优化目标函数

    Args:
        base_config_path: 基础配置文件路径（可选）
        n_epochs: 每次试验的训练轮数
        dataset: 数据集名称
        target: 目标属性

    Returns:
        objective: Optuna 优化目标函数
    """

    def objective(trial):
        """Optuna 目标函数 - 最小化验证集 MAE"""

        # 1. 定义超参数搜索空间

        # 模型架构参数
        alignn_layers = trial.suggest_int("alignn_layers", 2, 6)
        gcn_layers = trial.suggest_int("gcn_layers", 2, 6)
        hidden_features = trial.suggest_categorical("hidden_features", [128, 256, 512])
        embedding_features = trial.suggest_categorical("embedding_features", [32, 64, 128])
        edge_input_features = trial.suggest_categorical("edge_input_features", [40, 80, 120])
        triplet_input_features = trial.suggest_categorical("triplet_input_features", [20, 40, 60])

        # 训练参数
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # 正则化参数
        graph_dropout = trial.suggest_float("graph_dropout", 0.0, 0.5)

        # 注意力机制参数
        use_cross_modal_attention = trial.suggest_categorical("use_cross_modal_attention", [True, False])

        if use_cross_modal_attention:
            cross_modal_hidden_dim = trial.suggest_categorical("cross_modal_hidden_dim", [128, 256, 512])
            cross_modal_num_heads = trial.suggest_categorical("cross_modal_num_heads", [2, 4, 8])
            cross_modal_dropout = trial.suggest_float("cross_modal_dropout", 0.0, 0.3)
        else:
            cross_modal_hidden_dim = 256
            cross_modal_num_heads = 4
            cross_modal_dropout = 0.1

        use_fine_grained_attention = trial.suggest_categorical("use_fine_grained_attention", [True, False])

        if use_fine_grained_attention:
            fine_grained_num_heads = trial.suggest_categorical("fine_grained_num_heads", [4, 8, 16])
            fine_grained_dropout = trial.suggest_float("fine_grained_dropout", 0.0, 0.3)
        else:
            fine_grained_num_heads = 8
            fine_grained_dropout = 0.1

        # 中期融合参数
        use_middle_fusion = trial.suggest_categorical("use_middle_fusion", [True, False])

        if use_middle_fusion:
            # 选择在哪些层插入中期融合
            # 基于 alignn_layers 的值动态生成可选层
            max_layer = alignn_layers - 1  # 层索引从 0 开始

            # 建议的融合层（可以是单层或多层）
            # 为简单起见，我们从几个预定义选项中选择
            if alignn_layers >= 4:
                middle_fusion_layers_options = ["2", "1,3", "2,3", "1,2,3"]
            elif alignn_layers >= 3:
                middle_fusion_layers_options = ["1", "2", "1,2"]
            else:
                middle_fusion_layers_options = ["1"]

            middle_fusion_layers = trial.suggest_categorical("middle_fusion_layers", middle_fusion_layers_options)
            middle_fusion_hidden_dim = trial.suggest_categorical("middle_fusion_hidden_dim", [64, 128, 256])
            middle_fusion_num_heads = trial.suggest_categorical("middle_fusion_num_heads", [1, 2, 4])
            middle_fusion_dropout = trial.suggest_float("middle_fusion_dropout", 0.0, 0.3)
        else:
            middle_fusion_layers = "2"
            middle_fusion_hidden_dim = 128
            middle_fusion_num_heads = 2
            middle_fusion_dropout = 0.1

        # 2. 创建配置
        model_config = ALIGNNConfig(
            name="alignn",
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            hidden_features=hidden_features,
            embedding_features=embedding_features,
            edge_input_features=edge_input_features,
            triplet_input_features=triplet_input_features,
            graph_dropout=graph_dropout,
            use_cross_modal_attention=use_cross_modal_attention,
            cross_modal_hidden_dim=cross_modal_hidden_dim,
            cross_modal_num_heads=cross_modal_num_heads,
            cross_modal_dropout=cross_modal_dropout,
            use_fine_grained_attention=use_fine_grained_attention,
            fine_grained_num_heads=fine_grained_num_heads,
            fine_grained_dropout=fine_grained_dropout,
            use_middle_fusion=use_middle_fusion,
            middle_fusion_layers=middle_fusion_layers,
            middle_fusion_hidden_dim=middle_fusion_hidden_dim,
            middle_fusion_num_heads=middle_fusion_num_heads,
            middle_fusion_dropout=middle_fusion_dropout,
        )

        config = TrainingConfig(
            dataset=dataset,
            target=target,
            model=model_config,
            epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_early_stopping=20,  # 早停
            write_checkpoint=False,  # 不保存检查点以节省空间
            write_predictions=False,
            log_tensorboard=False,
            progress=False,  # 关闭进度条以减少输出
        )

        # 3. 加载数据
        try:
            train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
                dataset=config.dataset,
                target=config.target,
                n_train=config.n_train,
                n_val=config.n_val,
                n_test=config.n_test,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                batch_size=config.batch_size,
                atom_features=config.atom_features,
                neighbor_strategy=config.neighbor_strategy,
                standardize=False,
                id_tag=config.id_tag,
                pin_memory=config.pin_memory,
                workers=config.num_workers,
                save_dataloader=config.save_dataloader,
                use_canonize=config.use_canonize,
                filename=config.filename,
                cutoff=config.cutoff,
                max_neighbors=config.max_neighbors,
                output_dir=config.output_dir,
                classification_threshold=config.classification_threshold,
                target_multiplication_factor=config.target_multiplication_factor,
                standard_scalar_and_pca=config.standard_scalar_and_pca,
                keep_data_order=config.keep_data_order,
                output_features=config.model.output_features,
            )
        except Exception as e:
            print(f"Trial {trial.number} failed during data loading: {e}")
            raise optuna.TrialPruned()

        # 4. 创建模型
        model = ALIGNN(config.model)
        model.to(device)

        # 5. 定义损失函数和优化器
        if config.criterion == "mse":
            criterion = nn.MSELoss()
        elif config.criterion == "l1":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 6. 创建训练器和评估器
        trainer = create_supervised_trainer(
            model,
            optimizer,
            criterion,
            prepare_batch=prepare_batch,
            device=device,
        )

        evaluator = create_supervised_evaluator(
            model,
            metrics={
                "loss": Loss(criterion),
                "mae": MeanAbsoluteError(),
            },
            prepare_batch=prepare_batch,
            device=device,
        )

        # 7. 记录最佳验证 MAE
        best_val_mae = float('inf')

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            nonlocal best_val_mae

            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_mae = metrics["mae"]

            # 更新最佳 MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae

            # 报告中间值给 Optuna（用于剪枝）
            trial.report(val_mae, engine.state.epoch)

            # 检查是否应该剪枝
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 8. 训练模型
        try:
            trainer.run(train_loader, max_epochs=n_epochs)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed during training: {e}")
            raise optuna.TrialPruned()

        return best_val_mae

    return objective


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 Optuna 进行 ALIGNN 超参数调优")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 试验次数")
    parser.add_argument("--n_epochs", type=int, default=100, help="每次试验的训练轮数")
    parser.add_argument("--dataset", type=str, default="user_data", help="数据集名称")
    parser.add_argument("--target", type=str, default="target", help="目标属性")
    parser.add_argument("--output_dir", type=str, default="optuna_results", help="输出目录")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study 名称")
    parser.add_argument("--n_jobs", type=int, default=1, help="并行作业数（-1 表示使用所有 CPU）")
    parser.add_argument("--timeout", type=int, default=None, help="优化超时时间（秒）")
    parser.add_argument("--load_study", type=str, default=None, help="加载已有的 study 数据库路径")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 study 名称
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"alignn_optuna_{timestamp}"
    else:
        study_name = args.study_name

    # 创建或加载 study
    if args.load_study:
        storage = f"sqlite:///{args.load_study}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"加载已有 study: {study_name}")
    else:
        storage = f"sqlite:///{output_dir / 'optuna_study.db'}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            ),
        )
        print(f"创建新 study: {study_name}")

    # 创建目标函数
    objective = create_objective(
        n_epochs=args.n_epochs,
        dataset=args.dataset,
        target=args.target,
    )

    print("=" * 80)
    print("开始 Optuna 超参数优化")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"目标: {args.target}")
    print(f"试验次数: {args.n_trials}")
    print(f"每次试验轮数: {args.n_epochs}")
    print(f"输出目录: {output_dir}")
    print(f"并行作业数: {args.n_jobs}")
    print("=" * 80)

    # 运行优化
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n优化被用户中断")

    # 输出结果
    print("\n" + "=" * 80)
    print("优化完成!")
    print("=" * 80)

    # 获取最佳试验
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"\n统计信息:")
    print(f"  完成的试验: {len(complete_trials)}")
    print(f"  剪枝的试验: {len(pruned_trials)}")

    if len(complete_trials) > 0:
        print(f"\n最佳试验:")
        trial = study.best_trial
        print(f"  验证 MAE: {trial.value:.6f}")
        print(f"  参数:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # 保存最佳参数
        best_params_path = output_dir / "best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump({
                "best_value": trial.value,
                "best_params": trial.params,
                "trial_number": trial.number,
            }, f, indent=2)
        print(f"\n最佳参数已保存到: {best_params_path}")

        # 保存所有试验结果
        trials_df = study.trials_dataframe()
        trials_csv_path = output_dir / "all_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False)
        print(f"所有试验结果已保存到: {trials_csv_path}")

        # 生成优化历史图
        try:
            import optuna.visualization as vis

            # 优化历史
            fig_history = vis.plot_optimization_history(study)
            fig_history.write_html(str(output_dir / "optimization_history.html"))

            # 参数重要性
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html(str(output_dir / "param_importances.html"))

            # 并行坐标图
            fig_parallel = vis.plot_parallel_coordinate(study)
            fig_parallel.write_html(str(output_dir / "parallel_coordinate.html"))

            print(f"\n可视化图表已保存到: {output_dir}")
            print(f"  - optimization_history.html")
            print(f"  - param_importances.html")
            print(f"  - parallel_coordinate.html")
        except ImportError:
            print("\n提示: 安装 plotly 以生成可视化图表")
            print("      pip install plotly kaleido")
    else:
        print("\n没有完成的试验")

    print("=" * 80)


if __name__ == "__main__":
    main()
