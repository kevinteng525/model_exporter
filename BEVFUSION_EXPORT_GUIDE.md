# BEVFusion 模型导出指南

## 问题总结

在尝试导出 BEVFusion 模型时遇到多个问题：
1. **Metainfo 字段错误**：`Cannot set cam2img to be a field of data because cam2img is already a metainfo field`
2. **数据结构复杂性**：MMDetection3D 的 `Det3DDataSample` 包含复杂的嵌套结构
3. **必需参数缺失**：BEVFusion 需要 `data_samples` 参数，即使为空

## 关键发现

1. **数据格式要求**：
   - `inputs`: 包含 `points` (list[torch.Tensor]) 和可选的 `imgs` (torch.Tensor)
   - `data_samples`: Det3DDataSample 列表（必需，即使为空）

2. **Metainfo 限制**：
   - MMDetection 的 DataElement 保护某些字段不被修改
   - 深拷贝会包含所有字段，导致无法重建

3. **当前配置问题**：
   - 测试配置可能只包含 LiDAR 数据（points）
   - BEVFusion 通常需要多模态输入（points + imgs）

## 推荐解决方案

### 方案 1：使用完整数据（推荐）

```python
# 确保配置文件包含多模态数据
# 检查配置文件中的 data_sampler 配置
# 确保 dataset 包含 both LiDAR and camera data

# 使用改进的导出器
from improved_exporter import InputFlattener, ModelWrapper

class SafeModelWrapper(torch.nn.Module):
    def __init__(self, original_model, flattener):
        super().__init__()
        self.original_model = original_model
        self.flattener = flattener
        self.num_inputs = len(flattener.tensor_info)

    def forward(self, *args):
        flat_tensors = list(args)

        # 简单重建，只保留必要字段
        reconstructed = {}

        # 根据 tensor_info 重建 inputs
        for idx, tensor in enumerate(flat_tensors):
            if idx < len(self.flattener.tensor_info):
                path = self.flattener.tensor_info[idx]['path']

                if 'points' in path:
                    if 'points' not in reconstructed:
                        reconstructed['points'] = []
                    reconstructed['points'].append(tensor)
                elif 'imgs' in path:
                    reconstructed['imgs'] = tensor

        # 创建空的 data_samples
        data_samples = []

        return self.original_model(inputs=reconstructed, data_samples=data_samples, mode='predict')
```

### 方案 2：修改配置文件

检查并修改 BEVFusion 配置，确保使用多模态数据：

```python
# 在配置文件中添加或修改：
data_sampler = dict(
    type='MultiModalityDet3DSampler',
    collate_type='multi_modality_collate',
    use_img=True,  # 确保使用图像数据
    use_lidar=True,  # 确保使用LiDAR数据
)
```

### 方案 3：使用官方工具

考虑使用 MMDeploy 官方工具：
```bash
# 安装 MMDeploy
pip install mmdeploy

# 使用官方导出工具
python -m mmdeploy.tools.deploy \
    configs/mmdet3d/bevfusion/bevfusion.py \
    {YOUR_CONFIG_PATH} \
    {YOUR_CHECKPOINT_PATH} \
    --work-dir {OUTPUT_DIR}
```

## 测试命令

### 使用真实数据测试
```bash
# 确保有图像数据
python bevfusion_final_exporter.py \
    /path/to/bevfusion/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --device cuda:0
```

### 使用随机数据测试
```bash
python bevfusion_final_exporter.py \
    /path/to/bevfusion/config.py \
    --use-random-data \
    --device cpu
```

## 注意事项

1. **数据完整性**：BEVFusion 设计用于多模态融合，仅使用 LiDAR 数据可能无法正常工作
2. **CUDA 要求**：BEVFusion 的某些操作（如 bev_pool）需要 CUDA 支持
3. **内存需求**：多模态处理需要大量内存，建议使用 GPU

## 文件说明

- `bevfusion_exporter.py` - 早期尝试版本
- `bevfusion_final_exporter.py` - 最终版本，处理了基本需求
- `test_bevfusion_simple.py` - 简单测试工具

## 建议

1. **使用官方工具**：优先考虑使用 MMDeploy 官方导出工具
2. **完整数据**：确保使用包含 LiDAR 和图像的完整数据集
3. **逐步调试**：先在 CPU 上测试，成功后再使用 GPU
4. **参考文档**：查看 MMDetection3D 和 BEVFusion 官方文档

## 当前限制

1. 无法完全保留原始数据结构的非张量属性
2. 可能丢失一些必要的元数据
3. 导出的模型可能需要在后处理中重新格式化输入

这些问题主要源于 MMDetection3D 的设计限制，建议等待官方对模型导出的更好支持。