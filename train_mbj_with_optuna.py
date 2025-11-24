#!/usr/bin/env python3
"""
使用 Optuna 进行 MBJ Bandgap 超参数调优

这个脚本结合了 train_with_cross_modal_attention.py 和 Optuna 调优框架，
专门用于优化 mbj_bandgap 性质预测的超参数。

运行方式:
    # 基本用法（50次试验）
    python train_mbj_with_optuna.py --n_trials 50

    # 并行优化
    python train_mbj_with_optuna.py --n_trials 100 --n_jobs 4

    # 自定义数据路径
    python train_mbj_with_optuna.py --root_dir ../dataset/ --n_trials 50
"""

import os
import sys
import csv
import json
import argparse
import torch
import optuna
from optuna.trial import TrialState
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from config import TrainingConfig
from models.alignn import ALIGNNConfig, ALIGNN
from data import get_train_val_loaders
from torch import nn
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, MeanAbsoluteError

# 导入数据加载所需的库
from jarvis.core.atoms import Atoms
from transformers import AutoTokenizer, AutoModel
from tokenizers.normalizers import BertNormalizer
import numpy as np


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def load_dataset(cif_dir, id_prop_file, dataset, property_name):
    """加载本地数据集

    Args:
        cif_dir: CIF 文件目录
        id_prop_file: 描述文件路径 (description.csv)
        dataset: 数据集名称
        property_name: 属性名称

    Returns:
        dataset_array: 包含样本字典的列表
    """
    print(f"\n{'='*60}")
    print(f"加载数据集: {dataset} - {property_name}")
    print(f"CIF目录: {cif_dir}")
    print(f"描述文件: {id_prop_file}")
    print(f"{'='*60}\n")

    # 读取CSV文件
    with open(id_prop_file, 'r') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"总样本数: {len(data)}")

    # 文本归一化器
    norm = BertNormalizer(lowercase=False, strip_accents=True,
                         clean_text=True, handle_chinese_chars=True)

    # 加载词汇映射 - 智能路径查找
    possible_paths = [
        'vocab_mappings.txt',
        './vocab_mappings.txt',
        os.path.join(os.path.dirname(__file__), 'vocab_mappings.txt'),
        os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src/vocab_mappings.txt'),
        '../vocab_mappings.txt',
        '../../vocab_mappings.txt',
    ]

    vocab_file = None
    for path in possible_paths:
        if os.path.exists(path):
            vocab_file = path
            break

    if vocab_file is None:
        raise FileNotFoundError(
            "无法找到 vocab_mappings.txt 文件。请确保文件存在于正确位置。\n"
            f"尝试过的路径: {possible_paths}"
        )

    print(f"使用词汇映射文件: {vocab_file}")
    with open(vocab_file, 'r') as f:
        mappings = f.read().strip().split('\n')
    mappings = {m[0]: m[2:] for m in mappings}

    def normalize(text):
        text = [norm.normalize_str(s) for s in text.split('\n')]
        out = []
        for s in text:
            norm_s = ''
            for c in s:
                norm_s += mappings.get(c, ' ')
            out.append(norm_s)
        return '\n'.join(out)

    # 构建数据集
    dataset_array = []
    skipped = 0

    for j in tqdm(range(len(data)), desc="加载数据"):
        try:
            id = data[j][0]
            target = data[j][1]

            # 读取CIF文件
            cif_file = os.path.join(cif_dir, f'{id}.cif')
            if not os.path.exists(cif_file):
                raise FileNotFoundError(f"CIF文件不存在: {cif_file}")

            atoms = Atoms.from_cif(cif_file)
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
            if skipped <= 5:  # 只显示前5个错误
                print(f"跳过样本 {id}: {e}")

    print(f"\n成功加载: {len(dataset_array)} 样本")
    print(f"跳过: {skipped} 样本\n")

    return dataset_array


def create_mbj_objective(
    dataset_array,
    target_property="target",
    n_epochs=100,
    early_stopping=20,
    batch_size_options=None,
):
    """创建 MBJ Bandgap 优化目标函数

    Args:
        dataset_array: 预加载的数据集数组
        target_property: 目标属性名称（默认"target"）
        n_epochs: 每次试验的训练轮数
        early_stopping: 早停轮数
        batch_size_options: 批次大小选项列表

    Returns:
        objective: Optuna 优化目标函数
    """

    if batch_size_options is None:
        batch_size_options = [16, 32, 64]

    def objective(trial):
        """Optuna 目标函数 - 最小化验证集 MAE"""

        print(f"\n{'='*80}")
        print(f"Trial {trial.number} - MBJ Bandgap 超参数优化")
        print(f"{'='*80}\n")

        # ========== 1. 定义超参数搜索空间 ==========

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
        batch_size = trial.suggest_categorical("batch_size", batch_size_options)

        # 正则化参数
        graph_dropout = trial.suggest_float("graph_dropout", 0.0, 0.5)

        # 跨模态注意力参数（晚期融合）
        use_cross_modal_attention = trial.suggest_categorical("use_cross_modal_attention", [True, False])

        if use_cross_modal_attention:
            cross_modal_hidden_dim = trial.suggest_categorical("cross_modal_hidden_dim", [128, 256, 512])
            cross_modal_num_heads = trial.suggest_categorical("cross_modal_num_heads", [2, 4, 8])
            cross_modal_dropout = trial.suggest_float("cross_modal_dropout", 0.0, 0.3)
        else:
            cross_modal_hidden_dim = 256
            cross_modal_num_heads = 4
            cross_modal_dropout = 0.1

        # 细粒度注意力参数
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
            # 使用固定的搜索空间（Optuna 要求所有试验中相同参数的选项必须一致）
            # 定义所有可能的层组合
            middle_fusion_layers = trial.suggest_categorical("middle_fusion_layers",
                                                             ["1", "2", "1,2", "1,3", "2,3", "1,2,3"])

            # 验证所选层是否与 alignn_layers 兼容
            selected_layers = [int(x) for x in middle_fusion_layers.split(",")]
            max_valid_layer = alignn_layers - 1  # 层索引从 0 开始

            # 如果选择的层超出了可用层数，则跳过此试验
            if any(layer > max_valid_layer for layer in selected_layers):
                raise optuna.TrialPruned(f"Selected layers {middle_fusion_layers} incompatible with alignn_layers={alignn_layers}")

            middle_fusion_hidden_dim = trial.suggest_categorical("middle_fusion_hidden_dim", [64, 128, 256])
            middle_fusion_num_heads = trial.suggest_categorical("middle_fusion_num_heads", [1, 2, 4])
            middle_fusion_dropout = trial.suggest_float("middle_fusion_dropout", 0.0, 0.3)
        else:
            middle_fusion_layers = "2"
            middle_fusion_hidden_dim = 128
            middle_fusion_num_heads = 2
            middle_fusion_dropout = 0.1

        # ========== 2. 创建配置 ==========

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
            dataset="user_data",        # 使用本地数据
            target=target_property,     # 目标属性
            model=model_config,
            epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_early_stopping=early_stopping,
            write_checkpoint=False,
            write_predictions=False,
            log_tensorboard=False,
            progress=False,
        )

        # ========== 3. 加载数据 ==========

        try:
            print("加载数据...")
            train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
                dataset_array=dataset_array,  # 使用预加载的数据集
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
                id_tag=config.id_tag,
                pin_memory=config.pin_memory,
                workers=config.num_workers,
                save_dataloader=config.save_dataloader,
                use_canonize=config.use_canonize,
                filename=config.filename,
                cutoff=config.cutoff,
                max_neighbors=config.max_neighbors,
                output_dir=config.output_dir,
                target_multiplication_factor=config.target_multiplication_factor,
                standard_scalar_and_pca=config.standard_scalar_and_pca,
                keep_data_order=config.keep_data_order,
            )
            print(f"✓ 训练集: {len(train_loader.dataset)} 样本")
            print(f"✓ 验证集: {len(val_loader.dataset)} 样本")
            print(f"✓ 测试集: {len(test_loader.dataset)} 样本")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise optuna.TrialPruned()

        # ========== 4. 创建模型 ==========

        try:
            model = ALIGNN(config.model)
            model.to(device)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ 模型参数数量: {total_params:,}")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            raise optuna.TrialPruned()

        # ========== 5. 定义损失函数和优化器 ==========

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ========== 6. 创建训练器和评估器 ==========

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

        # ========== 7. 训练过程监控 ==========

        best_val_mae = float('inf')

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            nonlocal best_val_mae

            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_mae = metrics["mae"]

            if val_mae < best_val_mae:
                best_val_mae = val_mae

            # 每10个epoch打印一次
            if engine.state.epoch % 10 == 0:
                print(f"Epoch {engine.state.epoch}/{n_epochs} - Val MAE: {val_mae:.6f} (Best: {best_val_mae:.6f})")

            # 报告中间值给 Optuna（用于剪枝）
            trial.report(val_mae, engine.state.epoch)

            # 检查是否应该剪枝
            if trial.should_prune():
                raise optuna.TrialPruned()

        # ========== 8. 训练模型 ==========

        try:
            print("\n开始训练...")
            trainer.run(train_loader, max_epochs=n_epochs)
            print(f"✓ 训练完成 - 最佳验证 MAE: {best_val_mae:.6f}")
        except optuna.TrialPruned:
            print(f"✗ Trial {trial.number} 被剪枝")
            raise
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            raise optuna.TrialPruned()

        return best_val_mae

    return objective


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 Optuna 优化 MBJ Bandgap 预测")
    parser.add_argument("--root_dir", type=str, default="../dataset/", help="数据集根目录")
    parser.add_argument("--dataset", type=str, default="jarvis", help="数据集名称（jarvis/mp等）")
    parser.add_argument("--property", type=str, default="mbj_bandgap", help="目标性质名称")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 试验次数")
    parser.add_argument("--n_epochs", type=int, default=100, help="每次试验的训练轮数")
    parser.add_argument("--early_stopping", type=int, default=20, help="早停轮数")
    parser.add_argument("--output_dir", type=str, default="mbj_optuna_results", help="输出目录")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study 名称")
    parser.add_argument("--n_jobs", type=int, default=1, help="并行作业数（-1 表示使用 CPU）")
    parser.add_argument("--timeout", type=int, default=None, help="优化超时时间（秒）")
    parser.add_argument("--load_study", type=str, default=None, help="加载已有的 study 数据库路径")

    # Pruning 参数
    parser.add_argument("--pruner", type=str, default="median",
                        choices=["median", "hyperband", "successive_halving", "percentile", "patient", "none"],
                        help="Pruning 策略: median (稳定), hyperband (激进), successive_halving (更激进), percentile (基于百分位), patient (保守), none (不剪枝)")
    parser.add_argument("--pruner_startup_trials", type=int, default=5,
                        help="Pruner 启动试验数（在此之前不剪枝）")
    parser.add_argument("--pruner_warmup_steps", type=int, default=10,
                        help="Pruner 预热步数（每个试验的前N步不剪枝）")
    parser.add_argument("--pruner_interval_steps", type=int, default=1,
                        help="Pruner 检查间隔（每N步检查一次是否剪枝）")
    parser.add_argument("--percentile_pruner_percentile", type=float, default=25.0,
                        help="Percentile Pruner 的百分位阈值（0-100）")
    parser.add_argument("--patient_pruner_patience", type=int, default=3,
                        help="Patient Pruner 的耐心值（允许多少步无改善）")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 study 名称
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"mbj_bandgap_optuna_{timestamp}"
    else:
        study_name = args.study_name

    # 创建 Pruner
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
            interval_steps=args.pruner_interval_steps,
        )
        pruner_desc = f"MedianPruner (startup={args.pruner_startup_trials}, warmup={args.pruner_warmup_steps})"
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=args.n_epochs,
            reduction_factor=3,
        )
        pruner_desc = f"HyperbandPruner (max_resource={args.n_epochs}, reduction_factor=3)"
    elif args.pruner == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=4,
            min_early_stopping_rate=0,
        )
        pruner_desc = "SuccessiveHalvingPruner (reduction_factor=4)"
    elif args.pruner == "percentile":
        pruner = optuna.pruners.PercentilePruner(
            percentile=args.percentile_pruner_percentile,
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
            interval_steps=args.pruner_interval_steps,
        )
        pruner_desc = f"PercentilePruner (percentile={args.percentile_pruner_percentile}%, startup={args.pruner_startup_trials})"
    elif args.pruner == "patient":
        pruner = optuna.pruners.PatientPruner(
            wrapped_pruner=optuna.pruners.MedianPruner(
                n_startup_trials=args.pruner_startup_trials,
                n_warmup_steps=args.pruner_warmup_steps,
                interval_steps=args.pruner_interval_steps,
            ),
            patience=args.patient_pruner_patience,
        )
        pruner_desc = f"PatientPruner (patience={args.patient_pruner_patience}, wrapped=MedianPruner)"
    else:  # none
        pruner = optuna.pruners.NopPruner()
        pruner_desc = "NopPruner (不剪枝)"

    # 创建或加载 study
    if args.load_study:
        storage = f"sqlite:///{args.load_study}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"✓ 加载已有 study: {study_name}")
    else:
        storage = f"sqlite:///{output_dir / 'optuna_study.db'}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=pruner,
        )
        print(f"✓ 创建新 study: {study_name}")
        print(f"✓ Pruning 策略: {pruner_desc}")

    # 加载本地数据集
    print("\n" + "=" * 80)
    print("加载本地数据集")
    print("=" * 80)

    # 构建数据路径
    if args.dataset.lower() == 'jarvis':
        cif_dir = os.path.join(args.root_dir, f'jarvis/{args.property}/cif/')
        id_prop_file = os.path.join(args.root_dir, f'jarvis/{args.property}/description.csv')
    elif args.dataset.lower() == 'mp':
        cif_dir = os.path.join(args.root_dir, 'mp_2018_new/')
        id_prop_file = os.path.join(args.root_dir, 'mp_2018_new/mat_text.csv')
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    # 检查路径是否存在
    if not os.path.exists(cif_dir):
        raise FileNotFoundError(
            f"CIF目录不存在: {cif_dir}\n"
            f"请检查 --root_dir 和 --dataset 参数"
        )
    if not os.path.exists(id_prop_file):
        raise FileNotFoundError(
            f"描述文件不存在: {id_prop_file}\n"
            f"请检查 --root_dir 和 --property 参数"
        )

    # 加载数据
    dataset_array = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)

    if len(dataset_array) == 0:
        raise ValueError("数据集为空！请检查数据文件。")

    print(f"✓ 成功加载 {len(dataset_array)} 个样本\n")

    # 创建目标函数
    objective = create_mbj_objective(
        dataset_array=dataset_array,
        target_property="target",  # load_dataset 使用 "target" 键
        n_epochs=args.n_epochs,
        early_stopping=args.early_stopping,
    )

    print("\n" + "=" * 80)
    print("Optuna 超参数优化")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"目标性质: {args.property}")
    print(f"数据目录: {args.root_dir}")
    print(f"样本数量: {len(dataset_array)}")
    print(f"试验次数: {args.n_trials}")
    print(f"每次试验轮数: {args.n_epochs}")
    print(f"早停轮数: {args.early_stopping}")
    print(f"输出目录: {output_dir}")
    print(f"并行作业数: {args.n_jobs}")
    print(f"Pruning 策略: {args.pruner}")
    if args.pruner != "none":
        print(f"  {pruner_desc}")
    print("=" * 80 + "\n")

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
        print("\n⚠️  优化被用户中断")

    # 输出结果
    print("\n" + "=" * 80)
    print("优化完成!")
    print("=" * 80)

    # 获取试验统计
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"\n统计信息:")
    print(f"  完成的试验: {len(complete_trials)}")
    print(f"  剪枝的试验: {len(pruned_trials)}")

    if len(complete_trials) > 0:
        print(f"\n最佳试验 (MBJ Bandgap):")
        trial = study.best_trial
        print(f"  验证 MAE: {trial.value:.6f} eV")
        print(f"  参数:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # 保存最佳参数
        best_params_path = output_dir / "best_params_mbj.json"
        with open(best_params_path, 'w') as f:
            json.dump({
                "property": "mbj_bandgap",
                "best_value": trial.value,
                "best_params": trial.params,
                "trial_number": trial.number,
            }, f, indent=2)
        print(f"\n✓ 最佳参数已保存到: {best_params_path}")

        # 保存所有试验结果
        trials_df = study.trials_dataframe()
        trials_csv_path = output_dir / "all_trials_mbj.csv"
        trials_df.to_csv(trials_csv_path, index=False)
        print(f"✓ 所有试验结果已保存到: {trials_csv_path}")

        # 生成优化历史图
        try:
            import optuna.visualization as vis

            # 优化历史
            fig_history = vis.plot_optimization_history(study)
            fig_history.write_html(str(output_dir / "mbj_optimization_history.html"))

            # 参数重要性
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html(str(output_dir / "mbj_param_importances.html"))

            # 并行坐标图
            fig_parallel = vis.plot_parallel_coordinate(study)
            fig_parallel.write_html(str(output_dir / "mbj_parallel_coordinate.html"))

            print(f"\n✓ 可视化图表已保存到: {output_dir}")
            print(f"  - mbj_optimization_history.html")
            print(f"  - mbj_param_importances.html")
            print(f"  - mbj_parallel_coordinate.html")
        except ImportError:
            print("\n提示: 安装 plotly 以生成可视化图表")
            print("      pip install plotly kaleido")
    else:
        print("\n❌ 没有完成的试验")

    print("=" * 80)
    print("\n下一步: 使用最佳参数训练完整模型")
    print(f"  python train_with_best_params.py \\")
    print(f"      --best_params {output_dir}/best_params_mbj.json \\")
    print(f"      --epochs 500 \\")
    print(f"      --dataset user_data \\")
    print(f"      --target target")
    print()


if __name__ == "__main__":
    main()
