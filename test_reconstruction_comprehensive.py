#!/usr/bin/env python3
"""
全面的输入结构重建测试
覆盖各种边界情况和复杂场景
"""

import sys
import torch
import numpy as np
from typing import Any, Dict, List

# Import InputFlattener from improved_exporter.py
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')
from improved_exporter import InputFlattener


def compare_data_structure(orig, recon, path=""):
    """深度比较两个数据结构"""
    if isinstance(orig, torch.Tensor) and isinstance(recon, torch.Tensor):
        if orig.shape != recon.shape:
            print(f"  ✗ {path}: 形状不匹配 {orig.shape} vs {recon.shape}")
            return False
        if not torch.allclose(orig, recon, atol=1e-6):
            print(f"  ✗ {path}: 值不匹配")
            return False
        return True

    elif isinstance(orig, dict) and isinstance(recon, dict):
        # 检查键数量
        if len(orig) != len(recon):
            print(f"  ✗ {path}: 字典键数量不匹配 {len(orig)} vs {len(recon)}")
            return False

        # 检查每个键
        for key in orig:
            if key not in recon:
                print(f"  ✗ {path}: 缺少键 '{key}'")
                return False
            new_path = f"{path}.{key}" if path else key
            if not compare_data_structure(orig[key], recon[key], new_path):
                return False
        return True

    elif isinstance(orig, list) and isinstance(recon, list):
        # 检查列表长度
        if len(orig) != len(recon):
            print(f"  ✗ {path}: 列表长度不匹配 {len(orig)} vs {len(recon)}")
            return False

        # 检查每个元素
        for i, (o, r) in enumerate(zip(orig, recon)):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            if not compare_data_structure(o, r, new_path):
                return False
        return True

    else:
        # 比较其他类型
        if orig != recon:
            print(f"  ✗ {path}: 值不匹配 {orig} vs {recon}")
            return False
        return True


def test_case_1_basic_dict():
    """测试用例 1：基本的嵌套字典"""
    print("\n[Test 1] 基本嵌套字典")
    print("-" * 40)

    original = {
        'inputs': {
            'voxels': torch.randn(10, 5, 3),
            'metadata': {
                'num_points': torch.tensor([100]),
                'device': 'cuda:0'  # 非张量值
            }
        },
        'mode': 'tensor'  # 非张量值
    }

    # 预期结果：只有张量被提取
    expected_tensors = 2

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    assert len(flat_tensors) == expected_tensors, f"期望 {expected_tensors} 个张量，实际 {len(flat_tensors)}"

    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 只比较张量部分
    success = compare_data_structure(
        {k: v for k, v in original.items() if k in ['inputs']},
        reconstructed.get('inputs', {})
    )

    print(f"  张量数量: {len(flat_tensors)} (期望: {expected_tensors})")
    print(f"  重建结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_case_2_nested_lists():
    """测试用例 2：包含列表的结构"""
    print("\n[Test 2] 包含列表的结构")
    print("-" * 40)

    original = {
        'batches': [
            {
                'images': torch.randn(2, 3, 224, 224),
                'labels': torch.randint(0, 10, (2,))
            },
            {
                'images': torch.randn(3, 3, 224, 224),
                'labels': torch.randint(0, 10, (3,))
            }
        ],
        'global_info': torch.tensor([1.0, 2.0, 3.0])
    }

    expected_tensors = 5  # 2 images + 2 labels + 1 global_info

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    assert len(flat_tensors) == expected_tensors, f"期望 {expected_tensors} 个张量，实际 {len(flat_tensors)}"

    reconstructed = flattener.reconstruct_inputs(flat_tensors)
    success = compare_data_structure(original, reconstructed)

    print(f"  张量数量: {len(flat_tensors)} (期望: {expected_tensors})")
    print(f"  重建结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_case_3_mixed_types():
    """测试用例 3：混合数据类型"""
    print("\n[Test 3] 混合数据类型")
    print("-" * 40)

    original = {
        'inputs': {
            'data': torch.randn(5, 10),
            'mask': torch.ones(5, dtype=torch.bool),
            'ids': torch.arange(5)
        },
        'config': {
            'batch_size': 5,
            'device': 'cuda'
        },
        'extra': [
            torch.tensor([1.0]),
            "string_value",  # 非张量
            torch.tensor(2)
        ]
    }

    expected_tensors = 5  # data, mask, ids, extra[0], extra[2]

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    assert len(flat_tensors) == expected_tensors, f"期望 {expected_tensors} 个张量，实际 {len(flat_tensors)}"

    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 只检查有张量的部分
    test_original = {
        'inputs': original['inputs'],
        'extra': [original['extra'][0], original['extra'][2]]
    }
    test_recon = {
        'inputs': reconstructed.get('inputs', {}),
        'extra': reconstructed.get('extra', [])
    }
    # 确保 extra 列表长度正确
    while len(test_recon['extra']) < 2:
        test_recon['extra'].append(None)

    success = compare_data_structure(test_original, test_recon)

    print(f"  张量数量: {len(flat_tensors)} (期望: {expected_tensors})")
    print(f"  重建结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_case_4_empty_and_special():
    """测试用例 4：空张量和特殊情况"""
    print("\n[Test 4] 空张量和特殊情况")
    print("-" * 40)

    original = {
        'empty_tensor': torch.randn(0, 10),  # 空张量，应该被跳过
        'normal_tensor': torch.randn(5, 10),
        'scalar_tensor': torch.tensor(3.14),
        'nested': {
            'empty_list': [],
            'list_with_empty': [
                torch.randn(2, 2),
                torch.randn(0, 0),  # 空张量
                torch.randn(3, 3)
            ]
        }
    }

    expected_tensors = 3  # normal_tensor, scalar_tensor, 两个非空张量在 list_with_empty

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    assert len(flat_tensors) == expected_tensors, f"期望 {expected_tensors} 个张量，实际 {len(flat_tensors)}"

    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 创建预期的结构（跳过空张量）
    expected_recon = {
        'normal_tensor': original['normal_tensor'],
        'scalar_tensor': original['scalar_tensor'],
        'nested': {
            'list_with_empty': [
                original['nested']['list_with_empty'][0],
                original['nested']['list_with_empty'][2]
            ]
        }
    }

    success = compare_data_structure(expected_recon, reconstructed)

    print(f"  张量数量: {len(flat_tensors)} (期望: {expected_tensors})")
    print(f"  跳过的空张量: 2")
    print(f"  重建结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_case_5_deep_nesting():
    """测试用例 5：深度嵌套结构"""
    print("\n[Test 5] 深度嵌套结构")
    print("-" * 40)

    original = {
        'level1': {
            'level2': {
                'level3': {
                    'level4': {
                        'data': torch.randn(2, 3),
                        'indices': [
                            torch.tensor([0, 1, 2]),
                            torch.tensor([3, 4, 5]),
                            torch.tensor([6, 7, 8])
                        ]
                    }
                }
            }
        },
        'parallel': [
            [
                torch.randn(1),
                torch.randn(1)
            ],
            [
                torch.randn(2),
                torch.randn(2)
            ]
        ]
    }

    expected_tensors = 8  # data + 3 indices + 4 parallel tensors

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    assert len(flat_tensors) == expected_tensors, f"期望 {expected_tensors} 个张量，实际 {len(flat_tensors)}"

    reconstructed = flattener.reconstruct_inputs(flat_tensors)
    success = compare_data_structure(original, reconstructed)

    print(f"  张量数量: {len(flat_tensors)} (期望: {expected_tensors})")
    print(f"  最大嵌套深度: 5")
    print(f"  重建结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_case_6_sparse_indices():
    """测试用例 6：稀疏索引"""
    print("\n[Test 6] 稀疏索引（跳跃的索引）")
    print("-" * 40)

    original = {
        'sparse_list': [
            None,  # 索引 0
            torch.tensor([1]),  # 索引 1
            None,  # 索引 2
            None,  # 索引 3
            torch.tensor([5]),  # 索引 4
            None,  # 索引 5
            torch.tensor([7])   # 索引 6
        ],
        'nested_sparse': {
            'data': [
                [torch.tensor([1, 2]), None],  # [0][0], [0][1]
                None,  # [1]
                [torch.tensor([3, 4, 5])]  # [2][0]
            ]
        }
    }

    # 注意：实际数据中不会有 None，这里只是为了展示索引跳跃
    # 在真实场景中，空位置会被填充为合适的默认值

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证稀疏列表的长度正确
    sparse_len = len(reconstructed.get('sparse_list', []))
    nested_len = len(reconstructed.get('nested_sparse', {}).get('data', []))

    success = (sparse_len >= 7 and nested_len >= 3)

    print(f"  张量数量: {len(flat_tensors)}")
    print(f"  sparse_list 长度: {sparse_len} (期望 >= 7)")
    print(f"  nested_sparse.data 长度: {nested_len} (期望 >= 3)")
    print(f"  重建结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_case_7_mmdet3d_realistic():
    """测试用例 7：类 MMDetection3D 真实数据结构"""
    print("\n[Test 7] MMDetection3D 真实数据结构")
    print("-" * 40)

    # 模拟真实的 MMDetection3D 输入
    original = {
        'inputs': {
            'voxels': torch.randn(1000, 20, 5),
            'num_points': torch.randint(1, 20, (1000,)),
            'coors': torch.randint(0, 100, (1000, 3)),
            'img': torch.randn(6, 3, 960, 1280),  # 6 张图片
            'img_metas': [
                {
                    'img_shape': torch.tensor([960, 1280, 3]),
                    'pad_shape': torch.tensor([960, 1280, 3]),
                    'scale_factor': torch.tensor([1.0, 1.0, 1.0])
                } for _ in range(6)
            ]
        },
        'data_samples': [
            {
                'gt_bboxes_3d': torch.randn(10, 7),
                'gt_labels_3d': torch.randint(0, 10, (10,)),
                'gt_pts_semantic_mask': torch.randint(0, 20, (100000,))
            }
            for _ in range(2)  # 2 个样本
        ]
    }

    expected_tensors = 1 + 1 + 1 + 1 + (6 * 3) + (2 * 3)  # voxels, num_points, coors, img, 6*3 meta, 2*3 sample

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    reconstructed = flattener.reconstruct_inputs(flat_tensors)
    success = compare_data_structure(original, reconstructed)

    print(f"  张量数量: {len(flat_tensors)} (期望约: {expected_tensors})")
    print(f"  图片数量: 6")
    print(f"  样本数量: 2")
    print(f"  重建结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_case_8_edge_cases():
    """测试用例 8：边缘情况"""
    print("\n[Test 8] 边缘情况")
    print("-" * 40)

    test_cases = []

    # 测试 1: 只有非张量值
    test_cases.append({
        'name': '只有非张量值',
        'data': {'a': 1, 'b': 'text', 'c': [1, 2, 3]},
        'expected_tensors': 0
    })

    # 测试 2: 空结构
    test_cases.append({
        'name': '空结构',
        'data': {},
        'expected_tensors': 0
    })

    # 测试 3: 单个张量
    test_cases.append({
        'name': '单个张量',
        'data': torch.randn(3, 4),
        'expected_tensors': 1
    })

    # 测试 4: 只有一层列表
    test_cases.append({
        'name': '单层列表',
        'data': [torch.randn(i+1) for i in range(5)],
        'expected_tensors': 5
    })

    all_success = True
    for test_case in test_cases:
        flattener = InputFlattener()
        flat_tensors = flattener.analyze_and_flatten(test_case['data'])

        success = len(flat_tensors) == test_case['expected_tensors']
        print(f"  {test_case['name']}: {len(flat_tensors)} 张量 (期望: {test_case['expected_tensors']}) {'✓' if success else '✗'}")
        all_success = all_success and success

    return all_success


def main():
    """运行所有测试"""
    print("=" * 60)
    print("全面的输入结构重建测试套件")
    print("=" * 60)

    tests = [
        test_case_1_basic_dict,
        test_case_2_nested_lists,
        test_case_3_mixed_types,
        test_case_4_empty_and_special,
        test_case_5_deep_nesting,
        test_case_6_sparse_indices,
        test_case_7_mmdet3d_realistic,
        test_case_8_edge_cases
    ]

    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            print(f"  ✗ 测试异常: {e}")
            results.append((test_func.__name__, False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        test_display = test_name.replace("test_case_", "Test ")
        print(f"{test_display}: {status}")

    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试都通过了！重建功能非常健壮。")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，需要进一步检查。")


if __name__ == '__main__':
    main()