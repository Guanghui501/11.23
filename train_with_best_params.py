#!/usr/bin/env python3
"""
使用 Optuna 找到的最佳超参数进行完整训练

运行方式:
    python train_with_best_params.py --best_params optuna_results/best_params.json --epochs 500
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

from config import TrainingConfig
from models.alignn import ALIGNNConfig
from data import get_train_val_loaders
from models.alignn import ALIGNN
from torch import nn
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def train_with_best_params(
    best_params_path,
    n_epochs=500,
    dataset="user_data",
    target="target",
    output_dir="best_model_output",
    use_early_stopping=True,
    n_early_stopping=50,
):
    """使用最佳参数训练模型

    Args:
        best_params_path: 最佳参数 JSON 文件路径
        n_epochs: 训练轮数
        dataset: 数据集名称
        target: 目标属性
        output_dir: 输出目录
        use_early_stopping: 是否使用早停
        n_early_stopping: 早停轮数
    """

    # 1. 加载最佳参数
    print("=" * 80)
    print("加载最佳超参数")
    print("=" * 80)

    with open(best_params_path, 'r') as f:
        best_params_data = json.load(f)

    best_params = best_params_data["best_params"]
    best_val_mae = best_params_data.get("best_value", None)

    print(f"最佳验证 MAE (来自 Optuna): {best_val_mae}")
    print("\n最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # 2. 创建配置
    model_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=best_params["alignn_layers"],
        gcn_layers=best_params["gcn_layers"],
        hidden_features=best_params["hidden_features"],
        embedding_features=best_params["embedding_features"],
        edge_input_features=best_params["edge_input_features"],
        triplet_input_features=best_params["triplet_input_features"],
        graph_dropout=best_params["graph_dropout"],
        use_cross_modal_attention=best_params["use_cross_modal_attention"],
        cross_modal_hidden_dim=best_params.get("cross_modal_hidden_dim", 256),
        cross_modal_num_heads=best_params.get("cross_modal_num_heads", 4),
        cross_modal_dropout=best_params.get("cross_modal_dropout", 0.1),
        use_fine_grained_attention=best_params["use_fine_grained_attention"],
        fine_grained_num_heads=best_params.get("fine_grained_num_heads", 8),
        fine_grained_dropout=best_params.get("fine_grained_dropout", 0.1),
        use_middle_fusion=best_params.get("use_middle_fusion", False),
        middle_fusion_layers=best_params.get("middle_fusion_layers", "2"),
        middle_fusion_hidden_dim=best_params.get("middle_fusion_hidden_dim", 128),
        middle_fusion_num_heads=best_params.get("middle_fusion_num_heads", 2),
        middle_fusion_dropout=best_params.get("middle_fusion_dropout", 0.1),
    )

    config = TrainingConfig(
        dataset=dataset,
        target=target,
        model=model_config,
        epochs=n_epochs,
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        n_early_stopping=n_early_stopping if use_early_stopping else None,
        write_checkpoint=True,
        write_predictions=True,
        log_tensorboard=True,
        progress=True,
        output_dir=output_dir,
    )

    # 3. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_save_path = output_path / "config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config.dict(), f, indent=2, default=str)
    print(f"\n配置已保存到: {config_save_path}")

    # 4. 加载数据
    print("\n" + "=" * 80)
    print("加载数据")
    print("=" * 80)

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

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 5. 创建模型
    print("\n" + "=" * 80)
    print("创建模型")
    print("=" * 80)

    model = ALIGNN(config.model)
    model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 6. 定义损失函数和优化器
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

    # 学习率调度器
    if config.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )
    else:
        scheduler = None

    # 7. 创建训练器和评估器
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

    # 8. 添加处理器

    # 进度条
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    # TensorBoard 日志
    if config.log_tensorboard:
        tb_logger = TensorboardLogger(log_dir=str(output_path / "tb_logs"))

        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda x: {"loss": x},
        )

        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.COMPLETED,
            tag="validation",
            metric_names=["loss", "mae"],
            global_step_transform=global_step_from_engine(trainer),
        )

    # 检查点保存
    if config.write_checkpoint:
        checkpointer = Checkpoint(
            {"model": model, "optimizer": optimizer},
            DiskSaver(str(output_path / "checkpoints"), require_empty=False),
            filename_prefix="best",
            score_function=lambda engine: -engine.state.metrics["mae"],
            score_name="neg_mae",
            n_saved=3,
            global_step_transform=global_step_from_engine(trainer),
        )

        evaluator.add_event_handler(Events.COMPLETED, checkpointer)

    # 早停
    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=n_early_stopping,
            score_function=lambda engine: -engine.state.metrics["mae"],
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    # 记录训练历史
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "learning_rate": [],
    }

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics

        # 记录指标
        training_history["val_loss"].append(metrics["loss"])
        training_history["val_mae"].append(metrics["mae"])
        training_history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        print(f"\nEpoch {engine.state.epoch}/{config.epochs}")
        print(f"  验证 Loss: {metrics['loss']:.6f}")
        print(f"  验证 MAE:  {metrics['mae']:.6f}")
        print(f"  学习率:    {optimizer.param_groups[0]['lr']:.2e}")

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

    # 9. 开始训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    trainer.run(train_loader, max_epochs=config.epochs)

    # 10. 评估测试集
    print("\n" + "=" * 80)
    print("评估测试集")
    print("=" * 80)

    evaluator.run(test_loader)
    test_metrics = evaluator.state.metrics

    print(f"测试 Loss: {test_metrics['loss']:.6f}")
    print(f"测试 MAE:  {test_metrics['mae']:.6f}")

    # 11. 保存训练历史
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\n训练历史已保存到: {history_path}")

    # 12. 保存最终结果
    results = {
        "best_params": best_params,
        "optuna_best_val_mae": best_val_mae,
        "final_test_loss": test_metrics['loss'],
        "final_test_mae": test_metrics['mae'],
        "total_epochs": trainer.state.epoch,
    }

    results_path = output_path / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"最终结果已保存到: {results_path}")

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="使用 Optuna 最佳参数训练 ALIGNN 模型")
    parser.add_argument("--best_params", type=str, required=True, help="最佳参数 JSON 文件路径")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--dataset", type=str, default="user_data", help="数据集名称")
    parser.add_argument("--target", type=str, default="target", help="目标属性")
    parser.add_argument("--output_dir", type=str, default="best_model_output", help="输出目录")
    parser.add_argument("--no_early_stopping", action="store_true", help="禁用早停")
    parser.add_argument("--early_stopping_patience", type=int, default=50, help="早停轮数")

    args = parser.parse_args()

    train_with_best_params(
        best_params_path=args.best_params,
        n_epochs=args.epochs,
        dataset=args.dataset,
        target=args.target,
        output_dir=args.output_dir,
        use_early_stopping=not args.no_early_stopping,
        n_early_stopping=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
