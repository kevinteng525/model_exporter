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

    # 重建 - 使用内联定义
    def parse_path_component(component):
        if '[' in component and component.endswith(']'):
            base = component.split('[')[0]
            idx = int(component.split('[')[1].split(']')[0])
            return base, idx
        else:
            return component, None

    def set_nested_value(obj, path, value):
        keys = path.split('.')
        if not keys:
            return
        current = obj
        for key in keys[:-1]:
            base_key, idx = parse_path_component(key)
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
        base_key, idx = parse_path_component(keys[-1])
        if idx is not None:
            if base_key not in current:
                current[base_key] = []
            while len(current[base_key]) <= idx:
                current[base_key].append(None)
            current[base_key][idx] = value
        else:
            current[base_key] = value

    def reconstruct_inputs(tensor_list, tensor_info):
        inputs = {}
        for i, tensor in enumerate(tensor_list):
            if i < len(tensor_info):
                info = tensor_info[i]
                set_nested_value(inputs, info['path'], tensor)
        return inputs

    # 创建完整的 tensor_info
    full_tensor_info = []
    for i, info in enumerate(tensor_info):
        full_tensor_info.append({
            'path': info['path'],
            'shape': info['shape'],
            'dtype': tensors[i].dtype,
            'device': tensors[i].device
        })

    reconstructed = reconstruct_inputs(tensors, full_tensor_info)

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

    # 重建 - 使用内联定义
    def parse_path_component(component):
        if '[' in component and component.endswith(']'):
            base = component.split('[')[0]
            idx = int(component.split('[')[1].split(']')[0])
            return base, idx
        else:
            return component, None

    def set_nested_value(obj, path, value):
        keys = path.split('.')
        if not keys:
            return
        current = obj
        for key in keys[:-1]:
            base_key, idx = parse_path_component(key)
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
        base_key, idx = parse_path_component(keys[-1])
        if idx is not None:
            if base_key not in current:
                current[base_key] = []
            while len(current[base_key]) <= idx:
                current[base_key].append(None)
            current[base_key][idx] = value
        else:
            current[base_key] = value

    def reconstruct_inputs(tensor_list, tensor_info):
        inputs = {}
        for i, tensor in enumerate(tensor_list):
            if i < len(tensor_info):
                info = tensor_info[i]
                set_nested_value(inputs, info['path'], tensor)
        return inputs

    # 创建完整的 tensor_info
    full_tensor_info = []
    for i, info in enumerate(tensor_info):
        full_tensor_info.append({
            'path': info['path'],
            'shape': info['shape'],
            'dtype': tensors[i].dtype,
            'device': tensors[i].device
        })

    reconstructed = reconstruct_inputs(tensors, full_tensor_info)

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


def debug_test_case_det3d():
    """测试 MMDetection3D 的 Det3DDataSample 结构"""
    print("\n调试 Test: MMDetection3D Det3DDataSample")
    print("-" * 40)

    # 模拟 InstanceData 类
    class MockInstanceData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # 模拟 PointData 类
    class MockPointData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # 模拟 Det3DDataSample 类
    class MockDet3DDataSample:
        def __init__(self):
            self.gt_instances_3d = None
            self.pred_instances_3d = None
            self.gt_instances = None
            self.pred_instances = None
            self.gt_pts_seg = None
            self.pred_pts_seg = None

    # 创建复杂的 MMDetection3D 数据样本
    sample = MockDet3DDataSample()

    # 设置 3D 实例数据
    sample.gt_instances_3d = MockInstanceData(
        bboxes_3d=torch.randn(5, 7),
        labels_3d=torch.randint(0, 10, (5,)),
        scores_3d=torch.rand(5)
    )

    # 设置 2D 实例数据
    sample.gt_instances = MockInstanceData(
        bboxes=torch.randn(5, 4),
        labels=torch.randint(0, 10, (5,)),
        scores=torch.rand(5)
    )

    # 设置点云分割数据
    sample.gt_pts_seg = MockPointData(
        pts_semantic_mask=torch.randint(0, 20, (1000,)),
        pts_instance_mask=torch.randint(0, 50, (1000,))
    )

    # 创建包含 Det3DDataSample 的完整输入
    input_data = {
        'inputs': {
            'voxels': torch.randn(1000, 20, 5),
            'num_points': torch.randint(1, 20, (1000,)),
            'coors': torch.randint(0, 100, (1000, 3)),
        },
        'data_samples': [sample],  # 包装在列表中
        'batch_input_shape': (960, 1280),
        'device': 'cuda:0'  # 非张量
    }

    print("原始数据结构:")
    print(f"  inputs: {list(input_data['inputs'].keys())}")
    print(f"  data_samples: {len(input_data['data_samples'])} 个样本")
    if input_data['data_samples']:
        sample = input_data['data_samples'][0]
        print(f"    sample.gt_instances_3d.bboxes_3d: {sample.gt_instances_3d.bboxes_3d.shape}")
        print(f"    sample.gt_instances_3d.labels_3d: {sample.gt_instances_3d.labels_3d.shape}")
        print(f"    sample.gt_instances.bboxes: {sample.gt_instances.bboxes.shape}")
        print(f"    sample.gt_pts_seg.pts_semantic_mask: {sample.gt_pts_seg.pts_semantic_mask.shape}")

    # 展平数据 - 使用内联 InputFlattener 实现
    class InputFlattener:
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
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    tensors.extend(self._extract_tensors(value, new_path))
            elif isinstance(data, list) and not isinstance(data, str):
                for idx, item in enumerate(data):
                    new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                    tensors.extend(self._extract_tensors(item, new_path))
            elif hasattr(data, '__dict__') and id(data) not in self.processed_objects and len(dir(data)) > 10:
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

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(input_data)

    print(f"\n提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 验证预期结果
    expected_count = 11  # 3 inputs + 3 gt_instances + 2 gt_pts_seg + 1 gt_instances_3d + 2 non-tensor (batch_input_shape, device)
    actual_count = len(flat_tensors)

    print(f"\n期望张量数量: {expected_count}")
    print(f"实际张量数量: {actual_count}")

    # 检查关键张量是否被提取
    expected_paths = [
        'inputs.voxels',
        'inputs.num_points',
        'inputs.coors',
        'data_samples[0].gt_instances.bboxes',
        'data_samples[0].gt_instances.labels',
        'data_samples[0].gt_instances.scores',
        'data_samples[0].gt_instances_3d.bboxes_3d',
        'data_samples[0].gt_instances_3d.labels_3d',
        'data_samples[0].gt_pts_seg.pts_semantic_mask',
        'data_samples[0].gt_pts_seg.pts_instance_mask'
    ]

    success = True
    for path in expected_paths:
        found = any(info['path'] == path for info in flattener.tensor_info)
        if found:
            print(f"  ✓ {path} 已提取")
        else:
            print(f"  ✗ {path} 缺失")
            success = False

    print(f"\nDet3DDataSample 测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def debug_test_case_complex_objects():
    """测试复杂的对象嵌套"""
    print("\n调试 Test: 复杂对象嵌套")
    print("-" * 40)

    # 模拟复杂的嵌套对象结构
    class MockInstanceData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._private_attr = "should be skipped"
            self.method = lambda: "should be skipped"

    data = {
        'level1': {
            'tensor_data': torch.randn(3, 4),
            'object_data': {
                'inner_tensor': torch.randn(2, 3),
                'nested_object': MockInstanceData(
                    bbox=torch.randn(4),
                    label=torch.tensor(1),
                    confidence=0.95  # 非张量
                )
            },
            'list_of_objects': [
                MockInstanceData(points=torch.randn(10, 3)),
                MockInstanceData(points=torch.randn(15, 3)),
                MockInstanceData(points=torch.randn(20, 3))
            ]
        }
    }

    print("原始数据结构:")
    print(f"  level1: {list(data['level1'].keys())}")
    print(f"  level1.object_data: {list(data['level1']['object_data'].keys())}")
    print(f"  level1.list_of_objects: {len(data['level1']['list_of_objects'])} 个对象")

    # 展平 - 使用内联 InputFlattener 实现
    class InputFlattener:
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
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    tensors.extend(self._extract_tensors(value, new_path))
            elif isinstance(data, list) and not isinstance(data, str):
                for idx, item in enumerate(data):
                    new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                    tensors.extend(self._extract_tensors(item, new_path))
            elif hasattr(data, '__dict__') and id(data) not in self.processed_objects and len(dir(data)) > 10:
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

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(data)

    print(f"\n提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 预期的张量列表
    expected_paths = [
        'level1.tensor_data',
        'level1.object_data.inner_tensor',
        'level1.object_data.nested_object.bbox',
        'level1.object_data.nested_object.label',
        'level1.list_of_objects[0].points',
        'level1.list_of_objects[1].points',
        'level1.list_of_objects[2].points'
    ]

    print(f"\n预期张量数量: {len(expected_paths)}")
    print(f"实际张量数量: {len(flat_tensors)}")

    # 验证
    success = True
    for expected_path in expected_paths:
        found = any(info['path'] == expected_path for info in flattener.tensor_info)
        if found:
            print(f"  ✓ {expected_path} 已提取")
        else:
            print(f"  ✗ {expected_path} 缺失")
            success = False

    # 验证私有属性和方法被跳过
    private_found = any('_private_attr' in info['path'] for info in flattener.tensor_info)
    method_found = any('method' in info['path'] for info in flattener.tensor_info)

    if not private_found and not method_found:
        print("  ✓ 私有属性和方法正确被跳过")
    else:
        print("  ✗ 私有属性或方法被错误提取")
        success = False

    print(f"\n复杂对象嵌套测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def debug_test_case_edge_cases():
    """测试边缘情况"""
    print("\n调试 Test: 边缘情况")
    print("-" * 40)

    # 测试1: 空对象和非张量
    class EmptyObj:
        def __init__(self):
            self.non_tensor = "not a tensor"
            self.empty_tensor = torch.randn(0, 0)
            self.normal_tensor = torch.randn(2, 2)

    # 测试2: 循环引用（应该被避免）
    class CycleObj:
        def __init__(self):
            self.tensor = torch.randn(1)

    data1 = EmptyObj()
    cycle_obj1 = CycleObj()
    cycle_obj2 = CycleObj()
    cycle_obj1.reference = cycle_obj2
    cycle_obj2.reference = cycle_obj1  # 创建循环引用

    edge_cases = {
        'empty_obj': data1,
        'cycle_obj': cycle_obj1,
        'simple_tensor': torch.randn(3, 3),
        'non_tensor': "string",
        'boolean': True,
        'none': None,
        'int': 42,
        'float': 3.14
    }

    print("测试各种边缘情况:")
    for key, value in edge_cases.items():
        print(f"  {key}: {type(value).__name__}")

    # 展平 - 使用内联 InputFlattener 实现
    class InputFlattener:
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
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    tensors.extend(self._extract_tensors(value, new_path))
            elif isinstance(data, list) and not isinstance(data, str):
                for idx, item in enumerate(data):
                    new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                    tensors.extend(self._extract_tensors(item, new_path))
            elif hasattr(data, '__dict__') and id(data) not in self.processed_objects and len(dir(data)) > 10:
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

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(edge_cases)

    print(f"\n提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 验证结果
    expected_count = 4  # empty_obj.normal_tensor, cycle_obj.tensor, cycle_obj.reference.tensor, simple_tensor
    actual_count = len(flat_tensors)

    print(f"\n期望张量数量: {expected_count}")
    print(f"实际张量数量: {actual_count}")

    success = actual_count == expected_count
    if success:
        print("  ✓ 边缘情况处理正确")
    else:
        print("  ✗ 边缘情况处理有误")

    # 检查是否有循环引用错误（导致无限循环）
    if len(flattener.tensor_info) < 100:  # 合理的数量
        print("  ✓ 没有循环引用问题")
    else:
        print("  ✗ 可能存在循环引用问题")
        success = False

    print(f"\n边缘情况测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


if __name__ == '__main__':
    debug_test_case_1()
    debug_test_case_3()
    debug_test_case_det3d()
    debug_test_case_complex_objects()
    debug_test_case_edge_cases()