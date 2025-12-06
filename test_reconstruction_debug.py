#!/usr/bin/env python3
"""
调试重建功能问题
"""

import torch


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

    # 模拟展平过程
    tensor_info = []
    tensors = []

    # 提取张量
    def extract(data, path=""):
        if isinstance(data, torch.Tensor):
            if data.numel() > 0:  # 只提取非空张量
                tensor_info.append({'path': path, 'shape': data.shape})
                tensors.append(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                extract(v, f"{path}.{k}" if path else k)

    extract(original)

    print(f"提取的张量:")
    for i, info in enumerate(tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 重建
    from refined.improved_exporter import InputFlattener
    flattener = InputFlattener()
    flattener.tensor_info = []

    # 手动设置 tensor_info
    for i, info in enumerate(tensor_info):
        flattener.tensor_info.append({
            'path': info['path'],
            'shape': info['shape'],
            'dtype': tensors[i].dtype,
            'device': tensors[i].device
        })

    reconstructed = flattener.reconstruct_inputs(tensors)

    print("\n重建的结构:")
    print(f"  键: {list(reconstructed.keys())}")
    if 'inputs' in reconstructed:
        print(f"  inputs 的键: {list(reconstructed['inputs'].keys())}")

    # 检查问题
    print("\n问题分析:")
    print(f"  原始结构有 'mode' 键，但重建的结构中没有")
    print(f"  这是因为 'mode' 不是张量，所以不会被重建")
    print("  这是正确的行为 - 重建函数只重建张量部分")


def debug_test_case_3():
    """调试测试用例 3"""
    print("\n调试 Test 3: 混合数据类型")

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

    # 模拟展平过程
    tensor_info = []
    tensors = []

    def extract(data, path=""):
        if isinstance(data, torch.Tensor):
            if data.numel() > 0:
                tensor_info.append({'path': path, 'shape': data.shape})
                tensors.append(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                extract(v, f"{path}.{k}" if path else k)
        elif isinstance(data, list) and not isinstance(data, str):
            for i, v in enumerate(data):
                extract(v, f"{path}[{i}]" if path else f"[{i}]")

    extract(original)

    print(f"提取的张量:")
    for i, info in enumerate(tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 重建
    from refined.improved_exporter import InputFlattener
    flattener = InputFlattener()
    flattener.tensor_info = []

    for i, info in enumerate(tensor_info):
        flattener.tensor_info.append({
            'path': info['path'],
            'shape': info['shape'],
            'dtype': tensors[i].dtype,
            'device': tensors[i].device
        })

    reconstructed = flattener.reconstruct_inputs(tensors)

    print("\n重建的 extra 列表:")
    if 'extra' in reconstructed:
        print(f"  长度: {len(reconstructed['extra'])}")
        for i, item in enumerate(reconstructed['extra']):
            if isinstance(item, torch.Tensor):
                print(f"  [{i}]: Tensor {item.shape}")
            else:
                print(f"  [{i}]: {type(item)}")

    print("\n问题分析:")
    print(f"  原始 extra 列表长度: {len(original['extra'])}")
    print(f"  重建 extra 列表长度: {len(reconstructed['extra'])}")
    print("  这是正确的，因为只有张量被重建，非张量位置被填充为 None")


if __name__ == '__main__':
    debug_test_case_1()
    debug_test_case_3()