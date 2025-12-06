#!/usr/bin/env python3
"""
测试输入结构重建功能
"""

import torch

class InputFlattener:
    """简化的输入展平器，用于测试"""

    def __init__(self):
        self.tensor_info = []
        self.flatten_mapping = {}

    def analyze_and_flatten(self, data, path=""):
        """分析并展平输入数据"""
        self.tensor_info = []
        self.flatten_mapping = {}

        tensors = self._extract_tensors(data, path)

        # 创建反向映射
        for idx, info in enumerate(self.tensor_info):
            self.flatten_mapping[info['path']] = idx

        return tensors

    def _extract_tensors(self, data, path=""):
        """递归提取所有张量"""
        tensors = []

        if isinstance(data, torch.Tensor):
            # 找到张量
            info = {
                'path': path,
                'shape': data.shape,
                'dtype': data.dtype,
                'device': data.device
            }
            self.tensor_info.append(info)
            tensors.append(data)

        elif isinstance(data, dict):
            # 处理字典
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                tensors.extend(self._extract_tensors(value, new_path))

        elif isinstance(data, list) and not isinstance(data, str):
            # 处理列表
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                tensors.extend(self._extract_tensors(item, new_path))

        return tensors

    def _parse_path_component(self, component):
        """解析路径组件，返回 (key, index_or_none)"""
        if '[' in component and component.endswith(']'):
            # 处理列表索引，如 'data[0]'
            base = component.split('[')[0]
            idx = int(component.split('[')[1].split(']')[0])
            return base, idx
        else:
            # 处理普通键
            return component, None

    def _set_nested_value(self, obj, path, value):
        """在嵌套字典中设置值"""
        keys = path.split('.')

        if not keys:
            return

        current = obj

        # 处理除最后一个键之外的所有键
        for key in keys[:-1]:
            base_key, idx = self._parse_path_component(key)

            if idx is not None:
                # 处理列表索引
                if base_key not in current:
                    current[base_key] = []

                # 扩展列表到足够大小
                while len(current[base_key]) <= idx:
                    # 添加空字典，准备下一级
                    current[base_key].append({})

                current = current[base_key][idx]
            else:
                # 处理普通字典键
                if base_key not in current:
                    current[base_key] = {}
                current = current[base_key]

        # 处理最后一个键
        base_key, idx = self._parse_path_component(keys[-1])

        if idx is not None:
            # 处理列表索引赋值
            if base_key not in current:
                current[base_key] = []

            # 扩展列表到足够大小
            while len(current[base_key]) <= idx:
                current[base_key].append(None)

            # 设置值
            current[base_key][idx] = value
        else:
            # 处理普通字典键赋值
            current[base_key] = value

    def reconstruct_inputs(self, flat_tensors):
        """从展平的张量重建原始输入结构"""
        if not self.tensor_info:
            return {}

        # 创建重建后的输入字典
        inputs = {}

        for idx, tensor in enumerate(flat_tensors):
            if idx < len(self.tensor_info):
                info = self.tensor_info[idx]
                self._set_nested_value(inputs, info['path'], tensor)

        return inputs


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