# Auto-ASR（Gradio + 多后端 ASR + 字幕处理）

一个基于 Gradio 的ASR工具：上传/录制音频后进行转写，并导出字幕文件。支持多个转写后端，并提供一套可配置的字幕 LLM 处理流程（校正/翻译/智能断句）。

## 功能一览

- 转写后端
  - OpenAI 接口（远程）
  - FunASR（本地）
  - Transformers 后端（本地；当前默认用于 Qwen3-ASR）
- 长音频切分与字幕轴
  - 统一使用 **Silero VAD** 做切分/语音段时间轴
- 字幕处理（LLM）
  - 字幕校正
  - 字幕翻译
  - 智能断句

## 快速开始

1) 安装依赖

```bash
uv sync
```

2) 启动WebUI

```bash
uv run python app.py
```

启动后会打印本地访问地址（例如 `http://127.0.0.1:7860`），用浏览器打开即可。

3) 配置与持久化

- 网页界面中的配置会自动保存到项目根目录 `.auto_asr_config.json`
- 如需重置配置：删除 `.auto_asr_config.json`

## 可选：启用本地推理后端

### Qwen3-ASR（Transformers 本地推理）

安装：

```bash
uv sync --extra transformers
```

模型地址：

- Qwen3-ASR-1.7B：`https://huggingface.co/Qwen/Qwen3-ASR-1.7B`

使用方式：

- 在网页界面「引擎配置 -> Qwen3-ASR 本地推理」里点击「下载模型 / 加载模型」

### FunASR（本地推理）

安装：

```bash
uv sync --extra funasr
```

常用模型：

- SenseVoiceSmall：`https://huggingface.co/iic/SenseVoiceSmall`
- Fun-ASR-Nano-2512：`https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512`

## 切分与字幕轴（Silero VAD）

在网页界面「切分与字幕轴」里：

- `启用 VAD`：用于长音频切分、以及“语音段模式”的时间轴
- `时间轴策略`
  - `vad_speech`：优先按语音段输出字幕轴（推荐；对长音频更稳定）
  - `chunk`：按切分块输出字幕轴

## 字幕处理

在网页界面「字幕处理」标签页：

- 支持上传 `.srt/.vtt` 后做：
  - 字幕校正（LLM）
  - 字幕翻译（LLM）
  - 智能断句（LLM）
- LLM 配置独立于“转写引擎配置”，在「LLM 提供商（仅字幕处理）」区域配置即可
- 输出目录：默认写入 `./outputs/processed/`

## 模型与缓存目录

- 模型与下载缓存统一放在项目目录 `./models/` 下
- 具体子目录结构取决于下载器：
  - ModelScope：通常在 `./models/hub/models/<组织>/<模型名>/...`
  - HuggingFace：通常在 `./models/huggingface/hub/...`

## 常见问题

- `No module named 'typer'`（Gradio 依赖缺失）：
  - 先执行 `uv sync`，若仍报错可尝试：`uv pip install -U typer`
- Windows + Python 3.12 安装 FunASR 相关依赖失败（`llvmlite/numba`）：
  - 建议使用 Python 3.11（或 3.10）创建环境再安装：

```bash
uv python install 3.11
uv sync --extra funasr -p 3.11
```

- CUDA 加速（本地推理）：
  - 可按你的 CUDA 版本安装对应的 PyTorch，例如（CUDA 12.8）：

```bash
uv pip install --upgrade --torch-backend cu128 torch torchaudio
```

- ffmpeg：
  - 项目使用 `imageio-ffmpeg` 提供 ffmpeg

## Gradio 主题

- MIKU：`https://huggingface.co/spaces/NoCrypt/miku`
