# Subtitle LLM Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `auto-asr` 增加“字幕处理（LLM）”能力：对 `.srt/.vtt` 做字幕校正、翻译、分割，并在 Gradio WebUI 提供入口；复用现有 OpenAI Key/BaseURL 配置，但新增独立的 `LLM 模型名` 配置项。

**Architecture:** 新增 `subtitle_io`（解析/写回 SRT/VTT）+ `llm`（OpenAI Chat 封装、agent-loop 校验、可选磁盘缓存）+ `subtitle_processing`（Processor 抽象/注册表，translate/optimize/split 插件）+ `WebUI Tab`（上传字幕→选择处理器→导出文件）。UI 仅调用 pipeline，不直接写业务逻辑。

**Tech Stack:** Python, Gradio, openai SDK, pytest, ruff, `json-repair`（更稳地解析 LLM 输出 JSON）。

---

### Task 1: Subtitle IO（解析/写回）

**Files:**
- Create: `auto_asr/subtitle_io.py`
- Modify: `auto_asr/__init__.py`（导出可选）
- Test: `tests/test_subtitle_io.py`

**Step 1: Write the failing test**

```python
from auto_asr.subtitle_io import load_subtitle_file

def test_load_srt_basic(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\\n00:00:00,000 --> 00:00:01,000\\nhello\\n\\n", encoding="utf-8")
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].start_s == 0.0
    assert lines[0].end_s == 1.0
    assert lines[0].text == "hello"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_subtitle_io.py -q`
Expected: FAIL（模块/函数不存在）

**Step 3: Write minimal implementation**

- 支持 `.srt/.vtt`：
  - SRT 时间戳：`HH:MM:SS,mmm`
  - VTT 时间戳：`HH:MM:SS.mmm`（兼容 `WEBVTT` header）
- 输出统一为 `list[auto_asr.subtitles.SubtitleLine]`
- 只做必要容错：空行、序号行可选、`NOTE` 块跳过（VTT）

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_subtitle_io.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add auto_asr/subtitle_io.py tests/test_subtitle_io.py
git commit -m "feat: add subtitle io parser for srt/vtt"
```

---

### Task 2: LLM Client（OpenAI Chat + agent loop）

**Files:**
- Modify: `pyproject.toml`（增加 `json-repair` 依赖）
- Create: `auto_asr/llm/__init__.py`
- Create: `auto_asr/llm/client.py`
- Create: `auto_asr/llm/cache.py`（可选：磁盘缓存）
- Test: `tests/test_llm_utils.py`

**Step 1: Write the failing test**

```python
from auto_asr.llm.client import normalize_base_url

def test_normalize_base_url_adds_v1():
    assert normalize_base_url("https://api.openai.com") == "https://api.openai.com/v1"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_utils.py -q`
Expected: FAIL（模块不存在）

**Step 3: Write minimal implementation**

- `normalize_base_url()`：确保 base_url 以 `/v1` 结尾（参考 VideoCaptioner，但不引入其环境变量方案）
- `make_llm_client(api_key, base_url)`：复用 `auto_asr.openai_asr.make_openai_client` 或复制最小实现
- `call_chat_json_agent_loop()`：
  - 输入：system_prompt、payload dict、model、temperature、max_steps
  - 输出：dict（严格 key 完整性校验；失败则把 error 反馈回 messages 重试）
  - JSON 解析：`json_repair.loads`，并保证输出为 dict
- 可选磁盘缓存：`cache/llm/<sha>.json`，key=processor+model+prompt+payload hash

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_utils.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml auto_asr/llm tests/test_llm_utils.py
git commit -m "feat: add llm client with base url normalize and agent loop"
```

---

### Task 3: Subtitle Processing Core（Processor 抽象 + 注册表）

**Files:**
- Create: `auto_asr/subtitle_processing/__init__.py`
- Create: `auto_asr/subtitle_processing/base.py`
- Test: `tests/test_subtitle_processing_registry.py`

**Step 1: Write the failing test**

```python
from auto_asr.subtitle_processing.base import get_processor

def test_unknown_processor_raises():
    try:
        get_processor("nope")
        assert False
    except KeyError:
        assert True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_subtitle_processing_registry.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

- `SubtitleProcessor` 抽象类：`name`、`process(lines, cfg, *, llm) -> list[SubtitleLine]`
- `register_processor(cls)` 装饰器
- `get_processor(name)` / `list_processors()`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_subtitle_processing_registry.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add auto_asr/subtitle_processing tests/test_subtitle_processing_registry.py
git commit -m "feat: add subtitle processor base and registry"
```

---

### Task 4: Translate / Optimize / Split Processors（参考 VideoCaptioner 思路）

**Files:**
- Create: `auto_asr/subtitle_processing/processors/translate.py`
- Create: `auto_asr/subtitle_processing/processors/optimize.py`
- Create: `auto_asr/subtitle_processing/processors/split.py`
- Modify: `auto_asr/subtitle_processing/__init__.py`（导入注册）
- Test: `tests/test_subtitle_processors_split.py`

**Step 1: Write failing tests（split 最好先做：不依赖 OpenAI）**

```python
from auto_asr.subtitle_processing.processors.split import split_text_by_delimiter

def test_split_text_by_delimiter_basic():
    assert split_text_by_delimiter("a<br>b") == ["a", "b"]
```

**Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_subtitle_processors_split.py -q`
Expected: FAIL

**Step 3: Implement minimal processor logic**

- Translate:
  - 输入 batch: `{index: text}`
  - agent loop 校验 keys 完整一致
  - 输出替换 `SubtitleLine.text`（时间轴不变）
- Optimize:
  - agent loop 校验 keys 完整一致
  - 相似度阈值：短句更宽松；长句更严格（参考 VideoCaptioner 的 difflib 做法）
  - 校验失败降级：保留原文
- Split:
  - LLM 输出 `<br>` 分割（每条字幕逐条处理，支持并发）
  - 提供两种输出模式：
    - `inplace_newlines`：同一条字幕内用换行拼回（不改时间轴）
    - `split_to_cues`：拆成多条字幕，按字符/词数比例分配时间（保序、最小时长兜底）

**Step 4: Run tests**

Run: `uv run pytest tests/test_subtitle_processors_split.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add auto_asr/subtitle_processing/processors tests/test_subtitle_processors_split.py
git commit -m "feat: add subtitle processors (translate/optimize/split)"
```

---

### Task 5: Pipeline（文件→处理→导出）

**Files:**
- Create: `auto_asr/subtitle_processing/pipeline.py`
- Test: `tests/test_subtitle_processing_pipeline.py`

**Step 1: Write failing test**

```python
from auto_asr.subtitle_processing.pipeline import process_subtitle_file

def test_pipeline_requires_processor():
    try:
        process_subtitle_file("a.srt", processor="nope", out_dir=".")
        assert False
    except KeyError:
        assert True
```

**Step 2: Run**

Run: `uv run pytest tests/test_subtitle_processing_pipeline.py -q`
Expected: FAIL

**Step 3: Implement**

- `process_subtitle_file(in_path, *, processor, out_dir, options, openai_api_key, openai_base_url, llm_model, concurrency, batch_size)`
- 输出：处理后字幕路径 + 预览文本（前 N 条）+ debug 信息
- 输出路径：`outputs/processed/<stem>--<processor>--YYYYMMDD-HHMMSS.srt`

**Step 4: Run**

Run: `uv run pytest tests/test_subtitle_processing_pipeline.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add auto_asr/subtitle_processing/pipeline.py tests/test_subtitle_processing_pipeline.py
git commit -m "feat: add subtitle processing pipeline"
```

---

### Task 6: WebUI 接入（新增 Tab + 新增 LLM 模型名配置）

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_config_defaults.py`（只测默认值/保存字段存在）

**Step 1: Write failing test**

```python
from auto_asr.config import save_config, load_config

def test_config_roundtrip_includes_llm_model(tmp_path, monkeypatch):
    # 用 monkeypatch 覆盖 get_config_path 指向临时目录
    ...
```

**Step 2: Run**

Run: `uv run pytest tests/test_app_config_defaults.py -q`
Expected: FAIL

**Step 3: Implement**

- 在 OpenAI 配置块新增 `LLM 模型名（字幕处理）` textbox（默认 `gpt-4o-mini`）
- `_auto_save_settings()` 增加字段 `llm_model`
- 新增 Tab「字幕处理」：
  - 上传字幕文件（srt/vtt）
  - 处理类型（校正/翻译/分割）
  - 翻译目标语言
  - 分割模式（inplace_newlines / split_to_cues）
  - 并发数、batch_size（影响 LLM 请求）
  - 输出：预览、下载文件、debug

**Step 4: Run**

Run: `uv run pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py tests/test_app_config_defaults.py
git commit -m "feat: add subtitle processing tab and llm model config"
```

---

### Task 7: Docs + Verification

**Files:**
- Modify: `README.md`

**Steps:**
- 更新 README：新增“字幕处理（LLM）”说明、依赖、使用方式
- 代码格式与测试：
  - `uv run ruff format .`
  - `uv run ruff check .`
  - `uv run pytest`

**Commit:**

```bash
git add README.md
git commit -m "docs: add subtitle llm processing usage"
```

