#!/usr/bin/env python3
"""
独立的重建测试，使用 improved_exporter.py 中的 InputFlattener
"""

import sys
import torch

# Add mmdetection3d to path for accessing data structures
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')

# Import InputFlattener from improved_exporter.py
from improved_exporter import InputFlattener


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("基本功能测试")
    print("=" * 60)

    # 简单测试数据
    data = {
        'tensor1': torch.randn(10, 5),
        'nested': {
            'tensor2': torch.randn(20, 3),
            'tensor3': torch.randn(5, 5)
        },
        'list_data': [
            torch.randn(3, 3),
            torch.randn(3, 3)
        ]
    }

    # 测试展平
    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(data)

    print(f"原始数据: {len(flat_tensors)} 个张量")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 测试重建
    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证
    success = True
    if torch.allclose(data['tensor1'], reconstructed['tensor1']):
        print("✓ tensor1 重建成功")
    else:
        print("✗ tensor1 重建失败")
        success = False

    if torch.allclose(data['nested']['tensor2'], reconstructed['nested']['tensor2']):
        print("✓ nested.tensor2 重建成功")
    else:
        print("✗ nested.tensor2 重建失败")
        success = False

    if torch.allclose(data['nested']['tensor3'], reconstructed['nested']['tensor3']):
        print("✓ nested.tensor3 重建成功")
    else:
        print("✗ nested.tensor3 重建失败")
        success = False

    if (isinstance(reconstructed['list_data'], list) and
        len(reconstructed['list_data']) == 2 and
        torch.allclose(data['list_data'][0], reconstructed['list_data'][0]) and
        torch.allclose(data['list_data'][1], reconstructed['list_data'][1])):
        print("✓ list_data 重建成功")
    else:
        print("✗ list_data 重建失败")
        success = False

    print(f"\n基本功能测试结果: {'✅ 通过' if success else '❌ 失败'}")
    return success


def test_complex_structure():
    """测试复杂结构"""
    print("\n" + "=" * 60)
    print("复杂结构测试")
    print("=" * 60)

    # 复杂测试数据
    data = {
        'batch_inputs': {
            'points': [torch.randn(100, 5), torch.randn(200, 5), torch.randn(150, 5)],
            'features': torch.randn(3, 64, 32, 32),
            'metadata': {
                'batch_size': 3,
                'device': 'cuda:0'
            }
        },
        'data_samples': [
            {
                'gt_bboxes': torch.randn(10, 4),
                'gt_labels': torch.randint(0, 10, (10,)),
                'img_shape': (224, 224)
            },
            {
                'gt_bboxes': torch.randn(5, 4),
                'gt_labels': torch.randint(0, 10, (5,)),
                'img_shape': (224, 224)
            },
            {
                'gt_bboxes': torch.randn(8, 4),
                'gt_labels': torch.randint(0, 10, (8,)),
                'img_shape': (224, 224)
            }
        ]
    }

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(data)

    print(f"复杂数据: {len(flat_tensors)} 个张量")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 重建
    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证
    success = True

    # 检查 batch_inputs
    if 'batch_inputs' in reconstructed:
        if 'points' in reconstructed['batch_inputs']:
            points_recon = reconstructed['batch_inputs']['points']
            if isinstance(points_recon, list) and len(points_recon) == 3:
                for i, (orig, recon) in enumerate(zip(data['batch_inputs']['points'], points_recon)):
                    if torch.allclose(orig, recon):
                        print(f"✓ batch_inputs.points[{i}] 重建成功")
                    else:
                        print(f"✗ batch_inputs.points[{i}] 重建失败")
                        success = False
            else:
                print("✗ batch_inputs.points 结构错误")
                success = False
        else:
            print("✗ batch_inputs.points 缺失")
            success = False

        if 'features' in reconstructed['batch_inputs']:
            if torch.allclose(data['batch_inputs']['features'], reconstructed['batch_inputs']['features']):
                print("✓ batch_inputs.features 重建成功")
            else:
                print("✗ batch_inputs.features 重建失败")
                success = False
        else:
            print("✗ batch_inputs.features 缺失")
            success = False
    else:
        print("✗ batch_inputs 缺失")
        success = False

    # 检查 data_samples
    if 'data_samples' in reconstructed:
        data_samples_recon = reconstructed['data_samples']
        if isinstance(data_samples_recon, list) and len(data_samples_recon) == 3:
            for i in range(3):
                for key in ['gt_bboxes', 'gt_labels']:
                    if key in data_samples_recon[i]:
                        if torch.allclose(data['data_samples'][i][key], data_samples_recon[i][key]):
                            print(f"✓ data_samples[{i}].{key} 重建成功")
                        else:
                            print(f"✗ data_samples[{i}].{key} 重建失败")
                            success = False
                    else:
                        print(f"✗ data_samples[{i}].{key} 缺失")
                        success = False
        else:
            print("✗ data_samples 结构错误")
            success = False
    else:
        print("✗ data_samples 缺失")
        success = False

    print(f"\n复杂结构测试结果: {'✅ 通过' if success else '❌ 失败'}")
    return success


def test_empty_and_edge_cases():
    """测试空张量和边界情况"""
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)

    test_cases = [
        {
            'name': '包含空张量',
            'data': {
                'empty': torch.empty(0),
                'normal': torch.randn(5, 5)
            }
        },
        {
            'name': '深度嵌套',
            'data': {
                'level1': {
                    'level2': {
                        'level3': {
                            'deep_tensor': torch.randn(2, 2)
                        }
                    }
                }
            }
        },
        {
            'name': '混合类型',
            'data': {
                'tensor': torch.randn(3, 3),
                'string': 'test',
                'number': 42,
                'none': None,
                'list': [torch.randn(1, 1), 'text', 123]
            }
        }
    ]

    flattener = InputFlattener()
    all_success = True

    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")

        flat_tensors = flattener.analyze_and_flatten(test_case['data'])
        reconstructed = flattener.reconstruct_inputs(flat_tensors)

        print(f"  提取张量数: {len(flat_tensors)}")

        # 简单验证：检查是否没有丢失非空张量
        manual_count = 0
        def count_non_empty_tensors(data):
            count = 0
            if isinstance(data, torch.Tensor) and data.numel() > 0:
                count += 1
            elif isinstance(data, dict):
                for v in data.values():
                    count += count_non_empty_tensors(v)
            elif isinstance(data, list) and not isinstance(data, str):
                for item in data:
                    count += count_non_empty_tensors(item)
            return count

        expected_count = count_non_empty_tensors(test_case['data'])

        if len(flat_tensors) == expected_count:
            print(f"  ✓ 张量数量正确: {expected_count}")
        else:
            print(f"  ✗ 张量数量错误: 期望 {expected_count}, 实际 {len(flat_tensors)}")
            all_success = False

    print(f"\n边界情况测试结果: {'✅ 全部通过' if all_success else '❌ 部分失败'}")
    return all_success


def main():
    print("InputFlattener 独立测试")
    print("使用 improved_exporter.py 中的 InputFlattener")

    results = []
    results.append(test_basic_functionality())
    results.append(test_complex_structure())
    results.append(test_empty_and_edge_cases())

    print("\n" + "=" * 60)
    print("总体测试结果:")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有独立测试通过！")
        print("improved_exporter.py 中的 InputFlattener 功能完整。")
    else:
        print("❌ 部分测试失败！")
        print("需要进一步调试。")
    print("=" * 60)


if __name__ == '__main__':
    main()