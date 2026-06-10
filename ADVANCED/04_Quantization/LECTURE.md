# 04: Quantization — INT8/FP8 量化推理

## 0. 这条线是怎么连起来的

```
Step 01: FP32 FMA = 6.5 TFLOPS (Thor)
Step 03: INT8 IMMA = 46.7 TOPS

INT8 是 FP32 的 7x 理论算力！
但代价是精度下降。

你的问题：
  能不能把 AlexNet 的权重从 FP32 转成 INT8？
  推理时用 INT8 Tensor Core 加速？
  精度会掉多少？

这章回答这些问题。
```

## 1. 量化的本质

### 浮点数 → 整数的映射

```
FP32 数值范围:   ~3.4 × 10³⁸
INT8 数值范围:   -128 ~ 127 (有符号)  或 0 ~ 255 (无符号)

映射公式:
  Q = round(r / scale + zero_point)

  r:    原始浮点值
  Q:    量化后的整数
  scale:  缩放因子 (浮点)
  zero_point: 零点偏移 (整数)

反量化:
  r ≈ (Q - zero_point) × scale

直觉理解:
  scale 决定了一个 INT8 步长代表"多少"浮点值
  比如 scale = 0.1，那么 Q=0 → r=0.0, Q=1 → r=0.1, Q=127 → r=12.7
  zero_point 让 0.0 映射到某个整数（方便 padding 等操作）
```

### INT8 精度损失来源

```
量化误差 = |r - r_dequantized|

三种来源：
  1. 舍入误差: round() 会丢失信息（最多 ±0.5 step）
  2. 截断误差: 超出 [-128, 127] 的值被 clamp 掉
  3. 精度损失: 原来 FP32 能区分的微小差异，INT8 区分不了

例：
  weights = [0.0012, 0.0013]  (FP32)
  如果 scale=0.01:
    Q = round([0.12, 0.13]) = [0, 0]
  两个不同的权重被量化成了同一个值！
  → 这就是精度损失

  如何减小？scale 要选得合适。
```

## 2. 量化粒度

```
per-tensor (最粗，最简单):
  整个权重张量用同一个 scale + zero_point
  适用: 大部分层

per-channel (中等):
  每个输出通道有自己的 scale
  适用: Conv2d 的权重（每个卷积核的分布可能不同）
        Linear 的权重（每行的分布可能不同）

per-group (最细，GPTQ 等):
  每 G 个元素一组，每组有自己的 scale
  G=128 或 G=64
  适用: 大语言模型的权重（group size 越小精度越高，但存储 scale 的开销也大）

类比 Step 03 tiled matmul:
  Tiling = 把大矩阵分块，一块一块算
  per-group quantization = 把权重分块，一块一块缩放

  naive: 全用 FP32，不用量化 = 全部用一个 scale
  tiled: 分 tile 算 = per-channel/per-group
  WMMA: Tensor Core 加速 = INT8 Tensor Core
```

## 3. 校准 (Calibration)

### 什么是校准？

```
量化需要 scale。scale 怎么定？

思路：
  看一层权重的数值范围
  如果权重在 [-1.0, 1.0] 之间 → scale = (max - min) / 255

问题：
  只看权重不够，还要看激活值（activations）的分布
  激活值取决于输入数据

解决方法——校准：
  拿一批有代表性的输入（校准数据集）
  跑推理，收集每一层激活值的统计信息
  根据统计信息确定 scale
```

### 校准方法

| 方法 | 做法 | 精度 | 实现难度 |
|------|------|------|---------|
| Max (最大绝对值) | scale = max(abs(tensor)) / 127 | 低（outlier 会撑大 scale） | 最简单 |
| Percentile (百分位) | scale = percentile(abs(tensor), 99.9%) / 127 | 中（忽略极端离群值） | 简单 |
| KL 散度 | 找和原始分布 KL 散度最小的量化分布 | 高 | 中等 |
| MSE (均方误差) | 找最小化量化误差的 scale | 高 | 中等 |

### PyTorch 的校准流程

```python
import torch
import torch.ao.quantization as quant

# 1. 准备模型（插入 QuantStub/DeQuantStub）
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. 准备量化
model_prepared = quant.prepare(model)  # 插入 Observer，收集统计量

# 3. 校准
for images in calibration_dataloader:
    model_prepared(images)  # 每一层都记录激活值分布

# 4. 转换
model_quantized = quant.convert(model_prepared)
# 把权重换成 INT8，插入量化/反量化节点
# 推理时自动用 INT8 kernel
```

## 4. PTQ vs QAT

### PTQ (Post-Training Quantization)

```
步骤：
  1. 训练好一个 FP32 模型
  2. 用校准数据集跑一遍，收集统计量
  3. 直接量化权重和激活

优点：不需要重新训练
缺点：小模型（<10 MB）可能精度损失明显

适用：AlexNet、ResNet 等大模型，INT8 精度损失 < 1%
```

### QAT (Quantization-Aware Training)

```
步骤：
  1. 在训练过程中插入 FakeQuantize 节点
  2. FakeQuantize 在前向时模拟量化（量化→反量化）
  3. 反向传播时，梯度通过 FakeQuantize 的直通估计 (STE)
  4. 训练收敛后，转成真正的 INT8

优点：精度更高（模型学会了适应量化误差）
缺点：需要重新训练

适用：小模型、对精度要求高的场景
```

### PyTorch QAT 示例

```python
import torch.ao.quantization as quant

# 1. 定义模型时指定 qconfig
model.qconfig = quant.get_default_qat_qconfig('fbgemm')

# 2. 准备 QAT
model_qat = quant.prepare_qat(model, inplace=False)

# 3. 正常训练（FP32 forward, 但插入量化模拟）
for epoch in range(num_epochs):
    for images, labels in dataloader:
        loss = criterion(model_qat(images), labels)
        loss.backward()
        optimizer.step()

# 4. 转 INT8
model_int8 = quant.convert(model_qat)
```

## 5. PyTorch 量化 API 实战

### 模型结构要求

```
PyTorch 量化要求模型用 nn.Module 的子模块组合
不能直接用 nn.Sequential 套很多层

需要：
  - 用 QuantStub 标记输入量化点
  - 用 DeQuantStub 标记输出反量化点
  - 用 nn.Conv2d, nn.Linear 等标准层（quant 认识它们）
```

### 示例

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = quant.QuantStub()          # ← 输入量化
        self.conv = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 15 * 15, 10)
        self.dequant = quant.DeQuantStub()      # ← 输出反量化

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# PTQ 流程
model = MyModel().eval()
model.qconfig = quant.get_default_qconfig('qnnpack')
model_prepared = quant.prepare(model)
calibrate(model_prepared, calib_loader)     # 跑一遍校准
model_int8 = quant.convert(model_prepared)  # 转 INT8

# 推理
y = model_int8(x)  # 内部用 INT8 kernel
```

## 6. FP8 量化

### FP8 格式 (NVIDIA H100+, Thor)

```
FP8 有两种格式：
  E4M3 (4 位指数, 3 位尾数):  范围 [-448, 448]，精度高
  E5M2 (5 位指数, 2 位尾数):  范围 [-57344, 57344]，范围大，精度低

使用：
  前向: 权重用 E4M3，激活用 E5M2 (范围大)
  反向: 梯度用 E5M2
```

### FP8 和 INT8 的对比

```
INT8: 线性量化，没有指数
FP8:  浮点量化，有指数

对于大部分神经网络权重：
  FP8 比 INT8 精度更高（因为权重通常在 0 附近密集分布，FP8 在 0 附近的精度更高）
  但 FP8 Tensor Core 需要 H100+ 或 Thor+ 硬件

现状 (2025-2026):
  FP8 训练越来越成熟
  但我们的 Thor 测不了（前面 blocked：tcgen05 无法在 sm_110 使用）
```

## 7. TensorRT INT8

回顾 ADVANCED/02 的 TensorRT 部分：

```
TensorRT 的 INT8 优化比 PyTorch 更强：
  1. 自动做 kernel fusion（和 torch.compile 一样）
  2. 自动选 INT8 kernel（比 PyTorch 的量化更灵活）
  3. 可以做 INT8 + FP16 混合推理

工作流程：
  ONNX 导出 → TensorRT 读取 → INT8 calibration → 推理

INT8 calibration 在 TensorRT 里：
  trtexec --onnx=alexnet.onnx --int8 --calib=calib_images --saveEngine=alexnet.plan

  内部过程：
    1. 读取校准图片
    2. 每层跑 FP32，收集激活值分布
    3. 用 KL 散度找最优 INT8 scale
    4. 转 INT8 engine
```

## 8. 和前面步骤的连接

```
Step 03 INT8 IMMA = 46.7 TOPS:
  INT8 Tensor Core 的纯算力天花板
  量化推理就是要把你的模型算到这条线上
  int8_matmul_profile 里测出来的 46.7 TOPS

AlexNet 量化预期:
  FP32: 6.5 TFLOPS (纯 FP32 FMA)
  INT8: 理论上 46.7 TOPS

  实际加速不会到 7x，因为：
    - 量化/反量化有额外开销
    - memory-bound 的层不会因为 INT8 变快（因为瓶颈在带宽，不在算力）
    - 只有 compute-bound 的层才受益

  AlexNet 的 Conv 层 (AI=78-316): compute-bound → INT8 大幅加速
  AlexNet 的 FC 层 (AI=0.3-0.5): memory-bound → INT8 几乎无加速
  → 量化要配合瓶颈分类（Step 05）来做决策
```

## 9. 总结

```
量化 = 用更少比特存权重和激活 → 用 INT8 Tensor Core 算 → 更快 + 更省显存

流程:
  训练 FP32 → 校准 → 量化 → 推理 (INT8)

决策:
  compute-bound 的层 → 量化收益大（如 Conv, GEMM 大矩阵）
  memory-bound 的层 → 量化收益小（如 FC 小矩阵）
  精度敏感的层（如 attention 的 softmax 输入）→ 留在 FP16/FP32

和本课程的关系:
  你在 Step 03 测了 INT8 算力天花板
  在 alexnet_profile.py 看到 Conv 层 compute-bound
  量化 = 把 Conv 层推到 Step 03 测的 46.7 TOPS 上
```
