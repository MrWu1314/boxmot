# 视频剪辑说明

本项目基于 BoxMOT 多目标追踪框架，集成了人脸识别和追踪功能，用于视频中的人脸检测、追踪和自动剪辑。

## 环境安装

### 1. 安装 BoxMOT 依赖

```bash
pip install poetry
poetry install --with yolo
poetry shell
```

或者直接安装：

```bash
pip install boxmot
```

### 2. 安装人脸算法模型

本项目需要安装 DeepFace 0.0.93 版本用于人脸检测和属性分析：

```bash
git clone https://github.com/MrWu1314/deepface-0.0.93.git
cd deepface-0.0.93
pip install -e .
```

## 使用方法

### 快速开始

使用提供的测试脚本 `run_test.sh` 运行视频追踪：

```bash
bash run_test.sh
```

测试脚本的主要参数：
- `--yolo-model`: YOLO检测模型（默认：yolo11x.pt）
- `--reid-model`: ReID追踪模型（默认：osnet_x1_0_msmt17.pt）
- `--tracking-method`: 追踪方法（默认：botsort）
- `--source`: 输入视频路径
- `--classes`: 检测类别（0 表示人物）
- `--custom-save-path`: 输出保存路径
- `--device`: GPU设备ID

### 自定义参数运行

你也可以直接使用 Python 命令运行：

```bash
python tracking/track.py \
    --yolo-model yolo11x.pt \
    --reid-model osnet_x1_0_msmt17.pt \
    --tracking-method botsort \
    --source ./videos/your_video.mp4 \
    --classes 0 \
    --custom-save-path ./runs/custom_track \
    --device 0 \
    --min-age 1 \
    --max-age 100 \
    --max-face-angle 89
```

### 主要参数说明

- `--source`: 输入视频路径或摄像头（0表示摄像头）
- `--yolo-model`: YOLO检测模型路径
- `--reid-model`: ReID特征提取模型路径
- `--tracking-method`: 追踪算法（botsort/strongsort/bytetrack/ocsort等）
- `--classes`: 要追踪的目标类别（0=人物）
- `--custom-save-path`: 输出目录，包含：
  - `target_video/`: 每个追踪目标的单独视频
  - `custom_video.mp4`: 带追踪框的完整视频
  - `result.json`: 追踪结果JSON文件
- `--min-age` / `--max-age`: 目标年龄范围过滤
- `--max-face-angle`: 最大人脸角度阈值（用于过滤侧脸）

## 输出结果

运行完成后，在 `--custom-save-path` 指定的目录下会生成：

1. **target_video/**: 包含每个追踪目标的单独视频片段
2. **custom_video.mp4**: 带追踪框和ID标注的完整视频
3. **result.json**: 包含所有追踪信息的JSON文件

## 注意事项

- 确保已正确安装 DeepFace 0.0.93 版本
- 首次运行时会自动下载相应的模型权重
- 建议使用GPU加速以获得更好的性能
- 人脸角度和年龄过滤参数可根据实际需求调整
