#!/usr/bin/env python3
"""
测试 MMDetection3D 的 Det3DDataSample 结构重建
"""

import sys
import torch
import os

# Add mmdetection3d to path for accessing data structures
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')

# Import InputFlattener from improved_exporter.py
from improved_exporter import InputFlattener

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
        self.gt_instances_3d = MockInstanceData()
        self.gt_instances = MockInstanceData()
        self.gt_pts_seg = MockPointData()


def test_case_det3d_data_sample():
    """测试 Det3DDataSample 结构"""
    print("\n[Test] Det3DDataSample 结构")
    print("-" * 60)

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
        'data_samples': [sample],
        'batch_input_shape': (960, 1280),
        'device': 'cuda:0'
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

    # 使用 improved_exporter 中的 InputFlattener
    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(input_data)

    print(f"\n提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 重建数据
    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证重建结果
    print("\n验证重建结果:")
    success = True

    # 检查基本输入
    if 'inputs' in reconstructed:
        for key in ['voxels', 'num_points', 'coors']:
            if key in reconstructed['inputs']:
                orig = input_data['inputs'][key]
                recon = reconstructed['inputs'][key]
                if torch.allclose(orig, recon):
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

    print(f"\n测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def test_nested_mixed_types():
    """测试混合类型的嵌套结构"""
    print("\n[Test] 混合类型嵌套结构")
    print("-" * 60)

    # 创建包含张量和非张量的混合结构
    original_data = {
        'model_inputs': {
            'points': [torch.randn(100, 5), torch.randn(200, 5)],  # 张量列表
            'images': {
                'front': torch.randn(3, 224, 224),
                'back': torch.randn(3, 224, 224),
                'metadata': {
                    'camera_ids': ['front', 'back'],  # 非张量
                    'timestamp': '2023-01-01'  # 非张量
                }
            }
        },
        'model_config': {
            'voxel_size': [0.1, 0.1, 0.2],  # 非张量
            'point_cloud_range': [-50, -50, -5, 50, 50, 3]  # 非张量
        }
    }

    flattener = InputFlattener()
    flat_tensors = flattener.analyze_and_flatten(original_data)

    print(f"提取的张量:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: {info['shape']}")

    # 重建数据
    reconstructed = flattener.reconstruct_inputs(flat_tensors)

    # 验证重建结果
    print("\n验证重建结果:")
    success = True

    # 检查张量部分
    if 'model_inputs' in reconstructed:
        # 检查 points 列表
        if 'points' in reconstructed['model_inputs']:
            points_recon = reconstructed['model_inputs']['points']
            if isinstance(points_recon, list) and len(points_recon) == 2:
                for i, (orig, recon) in enumerate(zip(original_data['model_inputs']['points'], points_recon)):
                    if torch.allclose(orig, recon):
                        print(f"  ✓ model_inputs.points[{i}] 重建成功")
                    else:
                        print(f"  ✗ model_inputs.points[{i}] 重建失败")
                        success = False
            else:
                print("  ✗ model_inputs.points 结构不正确")
                success = False
        else:
            print("  ✗ model_inputs.points 缺失")
            success = False

        # 检查 images 字典
        if 'images' in reconstructed['model_inputs']:
            images_recon = reconstructed['model_inputs']['images']
            for view in ['front', 'back']:
                if view in images_recon:
                    if torch.allclose(original_data['model_inputs']['images'][view], images_recon[view]):
                        print(f"  ✓ model_inputs.images.{view} 重建成功")
                    else:
                        print(f"  ✗ model_inputs.images.{view} 重建失败")
                        success = False
                else:
                    print(f"  ✗ model_inputs.images.{view} 缺失")
                    success = False
        else:
            print("  ✗ model_inputs.images 缺失")
            success = False
    else:
        print("  ✗ model_inputs 缺失")
        success = False

    print(f"\n测试结果: {'✓ 成功' if success else '✗ 失败'}")
    return success


def main():
    print("=" * 60)
    print("InputFlattener MMDetection3D 测试")
    print("使用 improved_exporter.py 中的 InputFlattener")
    print("=" * 60)

    results = []
    results.append(test_case_det3d_data_sample())
    results.append(test_nested_mixed_types())

    print("\n" + "=" * 60)
    print("测试总结:")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有 MMDetection3D 测试通过！")
    else:
        print("❌ 部分测试失败，需要进一步调试。")
    print("=" * 60)


if __name__ == '__main__':
    main()