# 输入结构重建功能说明

本文档解释了模型导出工具中输入结构重建的工作原理和设计决策。

## 核心概念

### 1. 为什么需要重建？

MMDetection3D 模型接受复杂的嵌套数据结构作为输入：

```python
# 典型的 MMDetection3D 输入
inputs = {
    'inputs': {
        'voxels': torch.Tensor(1000, 20, 5),
        'num_points': torch.Tensor(1000),
        'coors': torch.Tensor(1000, 3),
    },
    'data_samples': [...]
}
```

而 `torch.export` 要求扁平的张量输入：

```python
# torch.export 需要的格式
def wrapped_model(tensor_0, tensor_1, tensor_2):
    # 重建原始结构
    inputs = {
        'inputs': {
            'voxels': tensor_0,
            'num_points': tensor_1,
            'coors': tensor_2
        }
    }
    return original_model(**inputs, mode='tensor')
```

### 2. 重建流程

```
原始数据 → 张量提取 → 扁平化 → 导出
    ↓
原始数据 ← 结构重建 ← 展平数据 ← 运行时
```

## 详细行为

### 1. 张量提取规则

- ✅ **提取**：所有非空的 `torch.Tensor`
- ❌ **跳过**：空张量 (`tensor.numel() == 0`)
- ❌ **跳过**：非张量类型（int、str、list、dict 等）

### 2. 路径记录

每个被提取的张量都会记录其在原始结构中的路径：

```python
# 原始结构
{
    'inputs': {
        'voxels': torch.Tensor(...),
        'data': [
            torch.Tensor(...),
            torch.Tensor(...)
        ]
    }
}

# 记录的路径
[
    'inputs.voxels',
    'inputs.data[0]',
    'inputs.data[1]'
]
```

### 3. 结构重建

重建时会：

1. **保留字典结构**：重建所有的字典层级
2. **保留列表索引**：保持列表元素的位置关系
3. **填充非张量位置**：用 `None` 填充被跳过的位置

### 4. 实际示例

#### 输入结构
```python
original = {
    'inputs': {
        'voxels': torch.randn(100, 5),      # 张量 0
        'mask': torch.ones(100),            # 张量 1
        'metadata': {                        # 非张量，跳过
            'device': 'cuda'
        }
    },
    'extra': [
        torch.tensor([1]),                  # 张量 2
        'text',                             # 非张量，跳过
        torch.tensor(2)                      # 张量 3
    ]
}
```

#### 提取结果
```python
tensor_info = [
    {'path': 'inputs.voxels', 'shape': (100, 5)},
    {'path': 'inputs.mask', 'shape': (100,)},
    {'path': 'extra[0]', 'shape': (1,)},
    {'path': 'extra[2]', 'shape': ()}
]

tensors = [tensor_0, tensor_1, tensor_2, tensor_3]
```

#### 重建结果
```python
reconstructed = {
    'inputs': {
        'voxels': tensor_0,
        'mask': tensor_1
        # metadata 被跳过
    },
    'extra': [
        tensor_0,      # 原 extra[0]
        None,          # 原 extra[1] 是字符串，填充 None
        tensor_3       # 原 extra[2]
    ]
}
```

## 设计决策

### 1. 为什么跳过非张量？

- **简化导出**：`torch.export` 只处理张量
- **减少复杂度**：避免处理 Python 对象
- **提高效率**：只传递必要的数据

### 2. 为什么保留列表索引？

- **保持语义**：`data[0]` 和 `data[2]` 的含义不同
- **避免错误**：防止后续代码期望特定索引
- **兼容性**：与原始数据结构保持兼容

### 3. 为什么用 `None` 填充？

- **明确标识**：清楚表明这是被跳过的位置
- **类型安全**：避免类型错误
- **易于检测**：可以轻松识别被填充的位置

## 使用建议

### 1. 导入模型后

```python
# 加载导出的模型
exported = torch.export.load('model.pt2')

# 准备输入（按 metadata.json 中的顺序）
tensor_0 = torch.randn(100, 5)
tensor_1 = torch.ones(100)
tensor_2 = torch.tensor([1])
tensor_3 = torch.tensor(2)

# 运行推理
output = exported(tensor_0, tensor_1, tensor_2, tensor_3)
```

### 2. 处理输出

输出中可能包含 `None` 值，需要相应处理：

```python
# 检查列表中的 None
for item in output.get('extra', []):
    if item is not None:
        # 处理张量
        process_tensor(item)
```

## 常见问题

### Q: 为什么重建的结构和原始不完全一样？
A: 因为只有张量被重建，非张量值被跳过。这是设计使然。

### Q: 列表长度为什么不匹配？
A: 被跳过的非张量位置用 `None` 填充，保持原始长度。

### Q: 空张量为什么被跳过？
A: 空张量不包含有效数据，跳过它们可以简化处理。

### Q: 如何知道哪些位置是填充的？
A: 查看 `metadata.json` 中的张量信息，或检查返回值中的 `None`。

## 测试验证

使用提供的测试脚本验证重建功能：

```bash
# 基础测试
python test_reconstruction.py

# 全面测试
python test_reconstruction_comprehensive.py

# 调试测试
python test_reconstruction_standalone.py
```

## 性能考虑

- **时间复杂度**：O(n)，n 是数据结构中的元素数量
- **空间复杂度**：O(m)，m 是张量数量
- **内存使用**：重建时会创建新的结构，但张量数据是共享的