#!/usr/bin/env python3
"""
BEVFusion 最终导出器

为 BEVFusion 提供简化的 data_samples 以满足模型要求
"""

import os
import sys
import argparse
import random
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner.utils import set_random_seed
from mmengine.logging import print_log
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample


# 用于导出的额外导入
import torch.onnx
from typing import Dict, List, Any, Tuple, Optional

from mmengine import init_default_scope

import sys

from mmengine.structures import InstanceData

# Add the mmdetection3d directory to the path (so projects can be imported)
mmdet3d_path = '/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d'
if mmdet3d_path not in sys.path:
    sys.path.insert(0, mmdet3d_path)

print("当前Python路径:")
for p in sys.path:
    print(f"  {p}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Export BEVFusion model')
    parser.add_argument('config', help='模型配置文件的路径')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='模型checkpoint路径（可选）')
    parser.add_argument(
        '--use-random-data',
        action='store_true',
        help='是否使用随机生成的数据代替真实数据集')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='推理和导出使用的设备')
    parser.add_argument(
        '--output-dir',
        default='./exported_models',
        help='导出模型保存目录')
    parser.add_argument(
        '--format',
        choices=['pt2', 'onnx', 'both'],
        default='both',
        help='导出格式')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细日志')
    args = parser.parse_args()
    return args


def to_device(data, device):
    """递归地将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [to_device(item, device) for item in data]
    else:
        return data


def get_real_dataloader_inputs(cfg, device):
    """使用 Runner 的类方法来构建数据加载器"""
    print("--- 1. 使用真实数据集加载输入 ---")
    print("正在使用 Runner 构建数据加载器...")

    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    print("数据加载器构建成功，正在获取第一个 batch...")
    data_batch = next(iter(dataloader))

    data_batch = to_device(data_batch, device)

    print("成功加载一个 batch 的真实数据。")
    return data_batch


def create_random_like(data):
    """递归地创建与输入数据结构相同的随机数据"""
    if isinstance(data, torch.Tensor):
        if torch.is_floating_point(data):
            return torch.randn_like(data)
        else:
            return torch.randint_like(data, low=0, high=10)
    elif isinstance(data, Mapping):
        return {key: create_random_like(value) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [create_random_like(item) for item in data]
    else:
        return data


def get_random_inputs(cfg, device):
    """生成与配置文件中定义的数据集具有相同结构和形状的随机输入"""
    print("--- 1. 使用随机数据生成输入 ---")
    print("首先，加载一个真实数据 batch 以获取其结构...")

    real_batch = get_real_dataloader_inputs(cfg, 'cpu')

    print("\n正在根据真实数据结构生成随机数据...")
    random_batch = create_random_like(real_batch)
    random_batch = to_device(random_batch, device)

    print("成功生成与真实数据结构相同的随机数据。")
    return random_batch


def extract_bevfusion_inputs(data):
    """提取 BEVFusion 输入并创建必要的张量列表"""
    # 提取 points
    points_list = []
    if 'inputs' in data and 'points' in data['inputs']:
        points_list = data['inputs']['points']
        print(f"提取到 {len(points_list)} 个 points 张量")

    # 提取 imgs
    img_tensor = None
    if 'inputs' in data and 'imgs' in data['inputs']:
        img_tensor = data['inputs']['imgs']
        print(f"提取到 imgs 张量")

    # 创建导出张量列表
    export_tensors = []
    tensor_info = []

    # 添加 points 张量
    for i, points in enumerate(points_list):
        export_tensors.append(points)
        tensor_info.append({
            'type': 'points',
            'index': i,
            'shape': points.shape,
            'dtype': points.dtype
        })

    # 添加 imgs 张量
    if img_tensor is not None:
        export_tensors.append(img_tensor)
        tensor_info.append({
            'type': 'imgs',
            'index': 0,
            'shape': img_tensor.shape,
            'dtype': img_tensor.dtype
        })

    return export_tensors, tensor_info, len(points_list)


class BEVFusionModelWrapper(torch.nn.Module):
    """BEVFusion 模型包装器"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,
                imgs: torch.Tensor,  # (B, N_cam, C, H, W)
                points: List[torch.Tensor],  # list[B] (N_i, 4)
                lidar2img: torch.Tensor,  # (B, N_cam, 4, 4)  必须 Tensor
                cam2img: torch.Tensor,  # (B, N_cam, 4, 4)
                ):
        B, N_CAM = imgs.shape[:2]

        inputs = dict(
            imgs=imgs,
            points=points,
        )

        # 2. 构造 data_samples，把外参挂到 metainfo
        data_samples: List[Det3DDataSample] = []
        for b in range(B):
            sample = Det3DDataSample()
            sample.gt_instances_3d = InstanceData()
            sample.gt_instances    = InstanceData()
            # 把张量拆成 numpy 再塞进去（框架里会再转回 np/torch）
            sample.set_metainfo(dict(
                img_shape=[(imgs.shape[-2], imgs.shape[-1])] * N_CAM,
                lidar2img=lidar2img[b].cpu().numpy(),   # (N_cam,4,4)
                cam2img=cam2img[b].cpu().numpy(),       # (N_cam,4,4)
                scale_factor=1.0,
                pad_shape=(imgs.shape[-2], imgs.shape[-1]),
            ))
            data_samples.append(sample)

        # 3. 走原 forward
        return self.model(inputs,
                               data_samples=data_samples,
                               mode='predict')


def export_to_pt2(model, inputs, output_path, verbose=False):
    """导出为 PT2 格式"""
    print(f"\n--- 导出 PT2 格式到 {output_path} ---")

    try:
        # 使用 torch.export 导出
        exported_program = torch.export.export(model, inputs)

        # 保存
        torch.export.save(exported_program, output_path)

        print(f"✅ PT2 导出成功！保存到: {output_path}")

        # 验证
        if verbose:
            print("验证 PT2 重新加载...")
            reloaded = torch.export.load(output_path)
            print("✅ PT2 重新加载验证成功！")

        return True

    except Exception as e:
        print(f"❌ PT2 导出失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def export_to_onnx(model, inputs, output_path, verbose=False):
    """导出为 ONNX 格式"""
    print(f"\n--- 导出 ONNX 格式到 {output_path} ---")

    try:
        # 准备输入名称
        input_names = []
        tensor_info = getattr(model, 'tensor_info', [])
        for info in tensor_info:
            if info['type'] == 'points':
                input_names.append(f"points_{info['index']}")
            else:
                input_names.append(info['type'])

        # 准备输出名称
        output_names = ['predictions']

        # 动态轴配置
        dynamic_axes = {}
        for input_name in input_names:
            if 'points' in input_name:
                dynamic_axes[input_name] = {0: 'num_points'}
            else:
                dynamic_axes[input_name] = {0: 'batch_size'}
        dynamic_axes['predictions'] = {0: 'batch_size'}

        # 导出
        torch.onnx.export(
            model,
            inputs,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )

        print(f"✅ ONNX 导出成功！保存到: {output_path}")

        # 验证
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX 模型验证成功！")
        except ImportError:
            print("⚠️  未安装 onnx，跳过验证")
        except Exception as e:
            print(f"⚠️  ONNX 验证失败: {e}")

        return True

    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def save_metadata(output_dir, config_path, checkpoint_path, tensor_info):
    """保存元数据"""
    metadata = {
        'model_type': 'BEVFusion',
        'model_config': config_path,
        'checkpoint': checkpoint_path,
        'tensor_info': tensor_info,
        'num_inputs': len(tensor_info),
        'note': 'BEVFusion 导出器，包含 points 和 imgs 输入'
    }

    metadata_path = Path(output_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"元数据已保存到: {metadata_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("BEVFusion 最终模型导出工具")
    print("=" * 60)

    # 1. 加载配置
    print("\n正在加载配置文件...")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = './work_dir'

    # 设置随机种子
    set_random_seed(0)

    # 2. 构建模型
    print("\n正在构建模型...")
    model = init_model(cfg,
                       args.checkpoint,
                       device=args.device)
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"模型构建成功，并已移动到 {device}，设置为 eval 模式。")

    # 3. 获取输入数据
    if args.use_random_data:
        inputs = get_random_inputs(cfg, device)
    else:
        inputs = get_real_dataloader_inputs(cfg, device)

    # 4. 提取 BEVFusion 输入
    print("\n--- 提取 BEVFusion 输入张量 ---")
    export_tensors, tensor_info, num_points = extract_bevfusion_inputs(inputs)

    print(f"\n成功提取 {len(export_tensors)} 个张量用于导出")
    for i, info in enumerate(tensor_info):
        print(f"  [{i}] {info['type']}[{info.get('index', 0)}]: shape={info['shape']}, dtype={info['dtype']}")

    # 5. 创建包装模型
    print("\n--- 创建 BEVFusion 包装模型 ---")
    wrapper_model = BEVFusionModelWrapper(model)
    wrapper_model = wrapper_model.to(device)
    wrapper_model.eval()

    B, N_CAM, C, H, W = 1, 6, 3, 256, 704
    example_imgs = torch.randn(B, N_CAM, C, H, W)
    example_points = [torch.randn(np.random.randint(20000, 30000), 4)
                      for _ in range(B)]
    # 外参：随便给单位矩阵，真实部署时换真标定
    example_lidar2img = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N_CAM, 1, 1)
    example_cam2img = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N_CAM, 1, 1)
    export_tensors = (example_imgs, example_points, example_lidar2img, example_cam2img)
    # 验证包装模型
    print("验证包装模型前向传播...")
    try:
        with torch.no_grad():
            outputs = wrapper_model(*export_tensors)
        print("✅ 包装模型前向传播成功！")
        print(f"输出类型: {type(outputs)}")
    except Exception as e:
        print("❌ 包装模型前向传播失败！")
        print(f"错误信息: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return

    # 6. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 7. 导出模型
    print("\n--- 导出模型 ---")

    success_count = 0

    if args.format in ['pt2', 'both']:
        pt2_path = output_dir / 'bevfusion_model.pt2'
        if export_to_pt2(wrapper_model, tuple(export_tensors), str(pt2_path), args.verbose):
            success_count += 1

    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / 'bevfusion_model.onnx'
        if export_to_onnx(wrapper_model, tuple(export_tensors), str(onnx_path), args.verbose):
            success_count += 1

    # 8. 保存元数据
    save_metadata(output_dir, args.config, args.checkpoint, tensor_info)

    # 9. 总结
    print("\n" + "=" * 60)
    print(f"导出完成！成功格式: {success_count}/{len(args.format.split(',')) if args.format != 'both' else 2}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()