#!/usr/bin/env python3
"""
直接测试 improved_exporter.py 中的 InputFlattener
"""

import sys
import torch

# Add mmdetection3d to path for accessing data structures
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')

# Import InputFlattener from improved_exporter.py
from improved_exporter import InputFlattener

def test_input_flattener():
    """测试 InputFlattener 类"""
    print("\n" + "=" * 60)
    print("测试 improved_exporter.py 中的 InputFlattener")
    print("=" * 60)

    # 测试数据 - 模拟 MMDetection3D 输入
    test_data = {
        'inputs': {
            'voxels': torch.randn(100, 20, 5),
            'num_points': torch.randint(1, 20, (100,)),
            'coors': torch.randint(0, 100, (100, 3)),
        },
        'data_samples': [
            {
                'gt_bboxes_3d': torch.randn(10, 7),
                'gt_labels_3d': torch.randint(0, 10, (10,)),
                'text_annotation': 'sample 0',  # 非张量
            },
            {
                'gt_bboxes_3d': torch.randn(5, 7),
                'gt_labels_3d': torch.randint(0, 10, (5,)),
                'text_annotation': 'sample 1',  # 非张量
            }
        ]
    }

    print("\n[1] 原始测试数据:")
    def print_structure(data, indent=0):
        prefix = '  ' * indent
        if isinstance(data, torch.Tensor):
            print(f"{prefix}Tensor: {data.shape}, {data.dtype}")
        elif isinstance(data, dict):
            for k, v in data.items():
                print(f"{prefix}{k}:")
                print_structure(v, indent + 1)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                print(f"{prefix}[{i}]:")
                print_structure(v, indent + 1)
        else:
            print(f"{prefix}{data}")

    print_structure(test_data)

    # 使用 InputFlattener
    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(test_data)

    print(f"\n[2] 展平结果:")
    print(f"提取到 {len(flat_tensors)} 个张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']} ({info['dtype']})")

    # 重建数据
    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    print(f"\n[3] 重建验证:")
    success = True

    # 检查 inputs 部分
    if 'inputs' in reconstructed:
        inputs_recon = reconstructed['inputs']
        for key in ['voxels', 'num_points', 'coors']:
            if key in inputs_recon:
                if torch.allclose(test_data['inputs'][key], inputs_recon[key]):
                    print(f"  ✓ inputs.{key} 重建成功")
                else:
                    print(f"  ✗ inputs.{key} 重建失败")
                    success = False
            else:
                print(f"  ✗ inputs.{key} 缺失")
                success = False
    else:
        print("  ✗ inputs 缺失")
        success = False

    # 检查 data_samples 部分
    if 'data_samples' in reconstructed:
        data_samples_recon = reconstructed['data_samples']
        if isinstance(data_samples_recon, list) and len(data_samples_recon) == 2:
            for i in range(2):
                for key in ['gt_bboxes_3d', 'gt_labels_3d']:
                    if key in data_samples_recon[i]:
                        if torch.allclose(test_data['data_samples'][i][key], data_samples_recon[i][key]):
                            print(f"  ✓ data_samples[{i}].{key} 重建成功")
                        else:
                            print(f"  ✗ data_samples[{i}].{key} 重建失败")
                            success = False
                    else:
                        print(f"  ✗ data_samples[{i}].{key} 缺失")
                        success = False
        else:
            print("  ✗ data_samples 结构不正确")
            success = False
    else:
        print("  ✗ data_samples 缺失")
        success = False

    print(f"\n[4] 测试结果:")
    if success:
        print("  ✅ 所有重建测试通过！")
        print("  improved_exporter.py 中的 InputFlattener 工作正常。")
    else:
        print("  ❌ 部分重建测试失败！")
        print("  请检查 InputFlattener 的实现。")

    print("=" * 60)
    return success


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试边界情况")
    print("=" * 60)

    test_cases = [
        {
            'name': '空张量',
            'data': {
                'empty': torch.empty(0),
                'normal': torch.randn(10, 5)
            }
        },
        {
            'name': '嵌套列表',
            'data': {
                'levels': [
                    [torch.randn(5, 3), torch.randn(5, 3)],
                    [torch.randn(5, 3)]
                ]
            }
        },
        {
            'name': '混合类型',
            'data': {
                'tensors': torch.randn(3, 3),
                'strings': ['a', 'b', 'c'],
                'numbers': [1, 2, 3],
                'nested': {
                    'tensor': torch.randn(2, 2),
                    'none_value': None
                }
            }
        }
    ]

    flattener = InputFlattener()
    all_success = True

    for i, test_case in enumerate(test_cases):
        print(f"\n[Test {i+1}] {test_case['name']}:")

        # 展平
        flat_tensors = flattener.analyze_and_flatten(test_case['data'])

        # 重建
        reconstructed = flattener.reconstruct_inputs(flat_tensors)

        # 简单验证：检查重建的张量数量是否正确
        expected_tensor_count = len(flat_tensors)

        # 手动计算期望的张量数量
        manual_count = 0
        def count_tensors(data):
            count = 0
            if isinstance(data, torch.Tensor) and data.numel() > 0:
                count += 1
            elif isinstance(data, dict):
                for v in data.values():
                    count += count_tensors(v)
            elif isinstance(data, list) and not isinstance(data, str):
                for item in data:
                    count += count_tensors(item)
            return count

        expected_manual = count_tensors(test_case['data'])

        if expected_tensor_count == expected_manual:
            print(f"  ✓ 张量数量正确: {expected_tensor_count}")
        else:
            print(f"  ✗ 张量数量错误: 期望 {expected_manual}, 实际 {expected_tensor_count}")
            all_success = False

    print(f"\n边界情况测试结果: {'✅ 全部通过' if all_success else '❌ 部分失败'}")
    return all_success


def main():
    print("InputFlattener 真实测试")
    print("测试 improved_exporter.py 中的 InputFlattener 类")

    result1 = test_input_flattener()
    result2 = test_edge_cases()

    print("\n" + "=" * 60)
    print("总体测试结果:")
    if result1 and result2:
        print("🎉 所有测试通过！")
        print("improved_exporter.py 中的 InputFlattener 可以正常工作。")
    else:
        print("❌ 部分测试失败！")
        print("需要检查 InputFlattener 的实现。")
    print("=" * 60)


if __name__ == '__main__':
    main()