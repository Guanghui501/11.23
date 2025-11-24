#!/usr/bin/env python
"""
快速检查检查点文件的格式和内容
用于调试模型加载问题
"""

import sys
from utils_retrieval import print_checkpoint_info

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_checkpoint.py <checkpoint_path>")
        print("\n示例:")
        print("  python check_checkpoint.py checkpoints/best_model.pt")
        print("  python check_checkpoint.py output_*/best_val_model.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    print_checkpoint_info(checkpoint_path)
