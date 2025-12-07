#!/usr/bin/env python3
"""
调试重建功能问题
"""

import sys
import torch

# Import InputFlattener from improved_exporter.py
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')
from improved_exporter import InputFlattener


def debug_test_case_1():
    """调试测试用例 1"""
    print("调试 Test 1: 基本嵌套字典")

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

    # 使用 improved_exporter 中的 InputFlattener
    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    print(f"提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 重建数据
    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证重建结果
    success = True
    if 'inputs' in reconstructed:
        if 'voxels' in reconstructed['inputs']:
            if torch.allclose(original['inputs']['voxels'], reconstructed['inputs']['voxels']):
                print("  ✓ inputs.voxels 重建成功")
            else:
                print("  ✗ inputs.voxels 重建失败")
                success = False
        else:
            print("  ✗ inputs.voxels 缺失")
            success = False

        if 'metadata' in reconstructed['inputs'] and 'num_points' in reconstructed['inputs']['metadata']:
            if torch.allclose(original['inputs']['metadata']['num_points'], reconstructed['inputs']['metadata']['num_points']):
                print("  ✓ inputs.metadata.num_points 重建成功")
            else:
                print("  ✗ inputs.metadata.num_points 重建失败")
                success = False
        else:
            print("  ✗ inputs.metadata.num_points 缺失")
            success = False
    else:
        print("  ✗ inputs 缺失")
        success = False

    print(f"测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def debug_test_case_2():
    """调试测试用例 2：复杂结构"""
    print("\n调试 Test 2: 复杂结构")

    original = {
        'data': [
            {
                'features': torch.randn(32, 64),
                'labels': torch.randint(0, 10, (32,))
            },
            {
                'features': torch.randn(16, 64),
                'labels': torch.randint(0, 10, (16,))
            }
        ],
        'global_info': {
            'mean': torch.randn(64),
            'std': torch.randn(64)
        }
    }

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    print(f"提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证重建结果
    success = True
    if 'data' in reconstructed and isinstance(reconstructed['data'], list):
        for i in range(2):
            if ('features' in reconstructed['data'][i] and
                torch.allclose(original['data'][i]['features'], reconstructed['data'][i]['features'])):
                print(f"  ✓ data[{i}].features 重建成功")
            else:
                print(f"  ✗ data[{i}].features 重建失败")
                success = False

            if ('labels' in reconstructed['data'][i] and
                torch.allclose(original['data'][i]['labels'], reconstructed['data'][i]['labels'])):
                print(f"  ✓ data[{i}].labels 重建成功")
            else:
                print(f"  ✗ data[{i}].labels 重建失败")
                success = False
    else:
        print("  ✗ data 结构重建失败")
        success = False

    if 'global_info' in reconstructed:
        for key in ['mean', 'std']:
            if (key in reconstructed['global_info'] and
                torch.allclose(original['global_info'][key], reconstructed['global_info'][key])):
                print(f"  ✓ global_info.{key} 重建成功")
            else:
                print(f"  ✗ global_info.{key} 重建失败")
                success = False
    else:
        print("  ✗ global_info 缺失")
        success = False

    print(f"测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def debug_test_case_3():
    """调试测试用例 3：空张量和特殊情况"""
    print("\n调试 Test 3: 空张量和特殊情况")

    original = {
        'empty_tensor': torch.empty(0),  # 空张量
        'normal_tensor': torch.randn(10, 5),
        'nested': {
            'another_empty': torch.zeros(0, 3),
            'valid_tensor': torch.ones(5, 5)
        }
    }

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original)

    print(f"提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 应该只提取非空张量
    expected_count = 2  # normal_tensor and valid_tensor
    if len(flat_tensors) == expected_count:
        print(f"  ✓ 正确跳过空张量 (提取了 {len(flat_tensors)} 个张量)")
    else:
        print(f"  ✗ 空张量处理错误 (期望 {expected_count} 个，实际 {len(flat_tensors)} 个)")

    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证只有非空张量被重建
    success = True
    if 'normal_tensor' in reconstructed:
        if torch.allclose(original['normal_tensor'], reconstructed['normal_tensor']):
            print("  ✓ normal_tensor 重建成功")
        else:
            print("  ✗ normal_tensor 重建失败")
            success = False
    else:
        print("  ✗ normal_tensor 缺失")
        success = False

    if 'nested' in reconstructed and 'valid_tensor' in reconstructed['nested']:
        if torch.allclose(original['nested']['valid_tensor'], reconstructed['nested']['valid_tensor']):
            print("  ✓ nested.valid_tensor 重建成功")
        else:
            print("  ✗ nested.valid_tensor 重建失败")
            success = False
    else:
        print("  ✗ nested.valid_tensor 缺失")
        success = False

    print(f"测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def main():
    print("=" * 60)
    print("InputFlattener 调试测试")
    print("使用 improved_exporter.py 中的 InputFlattener")
    print("=" * 60)

    results = []
    results.append(debug_test_case_1())
    results.append(debug_test_case_2())
    results.append(debug_test_case_3())

    print("\n" + "=" * 60)
    print("测试总结:")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有调试测试通过！")
    else:
        print("❌ 部分测试失败，需要进一步调试。")
    print("=" * 60)


if __name__ == '__main__':
    main()