#!/usr/bin/env python3
"""
测试输入结构重建功能
"""

import sys
import torch

# Import InputFlattener from improved_exporter.py
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')
from improved_exporter import InputFlattener


def test_nested_structure():
    """测试嵌套结构重建"""
    print("=" * 60)
    print("测试嵌套结构重建功能")
    print("=" * 60)

    # 创建复杂的嵌套结构
    original_data = {
        'inputs': {
            'voxels': torch.randn(100, 20, 5),
            'num_points': torch.randint(1, 20, (100,)),
            'coors': torch.randint(0, 100, (100, 3)),
        },
        'data_samples': [
            {
                'gt_bboxes_3d': torch.randn(10, 7),
                'gt_labels_3d': torch.randint(0, 10, (10,))
            },
            {
                'gt_bboxes_3d': torch.randn(5, 7),
                'gt_labels_3d': torch.randint(0, 10, (5,))
            }
        ]
    }

    print("\n[1] 原始数据结构:")
    def print_structure(data, indent=0):
        prefix = '  ' * indent
        if isinstance(data, torch.Tensor):
            print(f"{prefix}Tensor: {data.shape}")
        elif isinstance(data, dict):
            for k, v in data.items():
                print(f"{prefix}{k}:")
                print_structure(v, indent + 1)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                print(f"{prefix}[{i}]:")
                print_structure(v, indent + 1)
        else:
            print(f"{prefix}{type(data)}")

    print_structure(original_data)

    # 展平数据
    print("\n[2] 展平数据...")
    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original_data)

    print(f"提取到 {len(flat_tensors)} 个张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 重建数据
    print("\n[3] 重建数据结构...")
    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证重建结果
    print("\n[4] 验证重建结果...")
    success = True

    # 检查输入
    if 'inputs' in reconstructed:
        for key in ['voxels', 'num_points', 'coors']:
            if key in reconstructed['inputs']:
                orig = original_data['inputs'][key]
                recon = reconstructed['inputs'][key]
                if torch.allclose(orig, recon):
                    print(f"  ✓ inputs.{key} 重建成功")
                else:
                    print(f"  ✗ inputs.{key} 重建失败")
                    success = False
            else:
                print(f"  ✗ inputs.{key} 缺失")
                success = False

    # 检查 data_samples
    if 'data_samples' in reconstructed:
        data_samples = reconstructed['data_samples']
        if isinstance(data_samples, list) and len(data_samples) >= 2:
            for i in range(2):
                for key in ['gt_bboxes_3d', 'gt_labels_3d']:
                    if key in data_samples[i]:
                        orig = original_data['data_samples'][i][key]
                        recon = data_samples[i][key]
                        if torch.allclose(orig, recon):
                            print(f"  ✓ data_samples[{i}].{key} 重建成功")
                        else:
                            print(f"  ✗ data_samples[{i}].{key} 重建失败")
                            success = False
                    else:
                        print(f"  ✗ data_samples[{i}].{key} 缺失")
                        success = False

    # 总结
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！重建功能正常工作。")
    else:
        print("❌ 测试失败！重建功能存在问题。")
    print("=" * 60)


if __name__ == '__main__':
    test_nested_structure()