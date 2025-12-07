#!/usr/bin/env python3
"""
测试增强版的 InputFlattener
"""

import torch
import sys
import os

# 添加路径
sys.path.append('/refined')

# 直接复制 EnhancedInputFlattener 的实现
class EnhancedInputFlattener:
    def __init__(self):
        self.tensor_info = []
        self.flatten_mapping = {}
        self.processed_objects = set()

    def analyze_and_flatten(self, data, path=""):
        self.tensor_info = []
        self.flatten_mapping = {}
        self.processed_objects = set()
        tensors = self._extract_tensors(data, path)
        for idx, info in enumerate(self.tensor_info):
            self.flatten_mapping[info['path']] = idx
        return tensors

    def _extract_tensors(self, data, path=""):
        tensors = []
        if isinstance(data, torch.Tensor):
            if data.numel() > 0:
                info = {
                    'path': path,
                    'shape': data.shape,
                    'dtype': data.dtype,
                    'device': data.device
                }
                self.tensor_info.append(info)
                tensors.append(data)
            return tensors
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                tensors.extend(self._extract_tensors(value, new_path))
        elif isinstance(data, (list, tuple)) and not isinstance(data, str):
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                tensors.extend(self._extract_tensors(item, new_path))
        elif hasattr(data, '__dict__') and id(data) not in self.processed_objects:
            self.processed_objects.add(id(data))
            for attr_name in dir(data):
                if (attr_name.startswith('_') or
                    callable(getattr(data, attr_name, None)) or
                    isinstance(getattr(data, attr_name, None), type)):
                    continue
                try:
                    attr_value = getattr(data, attr_name)
                    new_path = f"{path}.{attr_name}" if path else attr_name
                    tensors.extend(self._extract_tensors(attr_value, new_path))
                except:
                    pass
        return tensors

    def _parse_path_component(self, component):
        if '[' in component and component.endswith(']'):
            base = component.split('[')[0]
            idx = int(component.split('[')[1].split(']')[0])
            return base, idx
        else:
            return component, None

    def _set_nested_value(self, obj, path, value):
        keys = path.split('.')
        if not keys:
            return
        current = obj
        for key in keys[:-1]:
            base_key, idx = self._parse_path_component(key)
            if idx is not None:
                if base_key not in current:
                    current[base_key] = []
                while len(current[base_key]) <= idx:
                    current[base_key].append({})
                current = current[base_key][idx]
            else:
                if base_key not in current:
                    current[base_key] = {}
                current = current[base_key]
        base_key, idx = self._parse_path_component(keys[-1])
        if idx is not None:
            if base_key not in current:
                current[base_key] = []
            while len(current[base_key]) <= idx:
                current[base_key].append(None)
            current[base_key][idx] = value
        else:
            current[base_key] = value

    def reconstruct_inputs(self, flat_tensors):
        if not self.tensor_info:
            return {}
        inputs = {}
        for idx, tensor in enumerate(flat_tensors):
            if idx < len(self.tensor_info):
                info = self.tensor_info[idx]
                self._set_nested_value(inputs, info['path'], tensor)
        return inputs

    def get_tensor_summary(self):
        summary = {
            'total_tensors': len(self.tensor_info),
            'tensor_paths': [info['path'] for info in self.tensor_info],
            'tensor_shapes': [info['shape'] for info in self.tensor_info],
            'tensor_dtypes': [str(info['dtype']) for info in self.tensor_info]
        }
        return summary


def test_enhanced_flattener():
    print("=" * 60)
    print("测试增强版 InputFlattener")
    print("=" * 60)

    # 测试1: Det3DDataSample 结构
    print("\n[Test 1] Det3DDataSample 结构")
    print("-" * 40)

    class MockInstanceData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MockDet3DDataSample:
        def __init__(self):
            self.gt_instances_3d = MockInstanceData(
                bboxes_3d=torch.randn(5, 7),
                labels_3d=torch.randint(0, 10, (5,))
            )
            self.gt_instances = MockInstanceData(
                bboxes=torch.randn(10, 4),
                labels=torch.randint(0, 80, (10,))
            )
            self._private = "should be skipped"
            self.method = lambda: None

    sample = MockDet3DDataSample()
    input_data = {
        'inputs': {
            'voxels': torch.randn(100, 20, 5),
            'num_points': torch.randint(1, 20, (100,)),
        },
        'data_samples': [sample],
    }

    flattener = EnhancedInputFlattener()
    flat_tensors = flattener.analyze_and_flatten(input_data)

    print(f"提取的张量数量: {len(flat_tensors)}")
    print("\n提取的张量路径:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}")

    # 验证关键路径
    expected_paths = [
        'inputs.voxels',
        'inputs.num_points',
        'data_samples[0].gt_instances.bboxes',
        'data_samples[0].gt_instances.labels',
        'data_samples[0].gt_instances_3d.bboxes_3d',
        'data_samples[0].gt_instances_3d.labels_3d'
    ]

    success = True
    for path in expected_paths:
        found = any(info['path'] == path for info in flattener.tensor_info)
        if found:
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ 缺失: {path}")
            success = False

    # 验证私有属性被跳过
    private_found = any('_private' in info['path'] for info in flattener.tensor_info)
    if not private_found:
        print("  ✓ 私有属性正确被跳过")
    else:
        print("  ✗ 私有属性被错误提取")
        success = False

    # 测试2: 复杂嵌套对象
    print("\n[Test 2] 复杂嵌套对象")
    print("-" * 40)

    class ComplexObject:
        def __init__(self):
            self.nested = {
                'tensor': torch.randn(3, 3),
                'deep': type('DeepObj', (), {
                    'tensor2': torch.randn(2, 2)
                })()
            }
            self._skip = "skip me"

    data = {
        'complex_obj': ComplexObject(),
        'simple_tensor': torch.randn(1)
    }

    flattener2 = EnhancedInputFlattener()
    flat_tensors2 = flattener2.analyze_and_flatten(data)

    print(f"提取的张量数量: {len(flat_tensors2)}")
    for i, info in enumerate(flattener2.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 测试3: 空张量处理
    print("\n[Test 3] 空张量和基本对象")
    print("-" * 40)

    class SimpleObj:
        def __init__(self):
            self.tensor = torch.randn(1)
            self.empty = torch.randn(0, 0)

    obj1 = SimpleObj()

    data3 = {
        'empty_tensor': torch.randn(0, 0),
        'normal_tensor': torch.randn(2, 2),
        'simple_obj': obj1,
        'none_val': None,
        'str_val': 'string'
    }

    flattener3 = EnhancedInputFlattener()
    flat_tensors3 = flattener3.analyze_and_flatten(data3)

    print(f"提取的张量数量: {len(flat_tensors3)} (期望: 2)")

    # 验证没有提取空张量
    empty_found = any(info['path'] == 'empty_tensor' for info in flattener3.tensor_info)

    if not empty_found:
        print("  ✓ 空张量正确被跳过")
    else:
        print("  ✗ 空张量被错误提取")
        success = False

    # 验证提取了正确的张量
    expected_paths = ['normal_tensor', 'simple_obj.tensor']
    for path in expected_paths:
        found = any(info['path'] == path for info in flattener3.tensor_info)
        if found:
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ 缺失: {path}")
            success = False

    # 汇总
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"Det3DDataSample 测试: {'✓ 通过' if success else '✗ 失败'}")

    return success


if __name__ == '__main__':
    test_enhanced_flattener()