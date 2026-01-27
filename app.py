from __future__ import annotations

import gradio as gr

from auto_asr.pipeline import transcribe_to_subtitles


def run_asr(
    audio_path: str | None,
    openai_api_key: str,
    openai_base_url: str,
    model: str,
    output_format: str,
    language: str,
    prompt: str,
    enable_vad: bool,
    vad_segment_threshold_s: int,
    vad_max_segment_threshold_s: int,
):
    if not audio_path:
        raise gr.Error("请先上传或录制一段音频。")
    if not (openai_api_key or "").strip():
        raise gr.Error("请先填写 OpenAI API Key。")

    lang = None if language == "auto" else language
    prompt = (prompt or "").strip() or None
    model = (model or "").strip() or "whisper-1"
    base_url = (openai_base_url or "").strip() or None

    try:
        result = transcribe_to_subtitles(
            input_audio_path=audio_path,
            openai_api_key=openai_api_key,
            openai_base_url=base_url,
            output_format=output_format,
            model=model,
            language=lang,
            prompt=prompt,
            enable_vad=enable_vad,
            vad_segment_threshold_s=int(vad_segment_threshold_s),
            vad_max_segment_threshold_s=int(vad_max_segment_threshold_s),
        )
    except Exception as e:
        raise gr.Error(f"转写失败：{e}") from e

    return result.preview_text, result.full_text, result.subtitle_file_path, result.debug


with gr.Blocks(title="auto-asr（OpenAI 转字幕）", theme=gr.themes.Ocean()) as demo:
    gr.Markdown(
        "\n".join(
            [
                "# auto-asr 音频转字幕",
                "上传/录制音频 -> OpenAI ASR -> 导出 SRT / VTT / TXT。",
                "",
                "- API 配置在页面中填写，不依赖环境变量。",
                "- 长音频自动切分：内置切分算法（源自 Qwen3-ASR-Toolkit，MIT）。",
                "- 如需真正的 VAD（Silero），安装：`uv sync --extra vad`（体积大，会拉 PyTorch）。",
            ]
        )
    )

    with gr.Accordion("OpenAI 配置", open=True):
        openai_api_key = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )
        openai_base_url = gr.Textbox(
            label="Base URL（可选）",
            placeholder="例如：https://api.openai.com/v1",
        )
        model = gr.Textbox(
            label="模型（默认 whisper-1）",
            value="whisper-1",
        )

    with gr.Row():
        audio_in = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="音频",
        )

    with gr.Row():
        output_format = gr.Dropdown(
            choices=[
                ("SRT 字幕", "srt"),
                ("VTT 字幕", "vtt"),
                ("纯文本", "txt"),
            ],
            value="srt",
            label="输出格式",
        )
        language = gr.Dropdown(
            choices=[
                ("自动检测", "auto"),
                ("中文", "zh"),
                ("英语", "en"),
                ("日语", "ja"),
                ("韩语", "ko"),
                ("法语", "fr"),
                ("德语", "de"),
                ("西语", "es"),
                ("俄语", "ru"),
            ],
            value="auto",
            label="语言",
        )

    prompt = gr.Textbox(
        label="提示词（可选）",
        placeholder="可填写术语/人名/地名等上下文，提升识别效果。",
    )

    with gr.Accordion("长音频切分", open=True):
        enable_vad = gr.Checkbox(value=True, label="启用切分（音频 >= 180 秒时生效）")
        vad_segment_threshold_s = gr.Slider(
            minimum=30,
            maximum=240,
            value=120,
            step=10,
            label="目标分段时长（秒）",
        )
        vad_max_segment_threshold_s = gr.Slider(
            minimum=60,
            maximum=360,
            value=180,
            step=10,
            label="最大分段时长（秒）",
        )

    run_btn = gr.Button("开始转写")

    with gr.Row():
        preview = gr.Textbox(label="字幕预览（前约 5000 字符）", lines=12)
    with gr.Row():
        full_text = gr.Textbox(label="完整文本", lines=12)
    with gr.Row():
        out_file = gr.File(label="下载")
        debug = gr.Textbox(label="调试信息", lines=2)

    run_btn.click(
        fn=run_asr,
        inputs=[
            audio_in,
            openai_api_key,
            openai_base_url,
            model,
            output_format,
            language,
            prompt,
            enable_vad,
            vad_segment_threshold_s,
            vad_max_segment_threshold_s,
        ],
        outputs=[preview, full_text, out_file, debug],
    )


if __name__ == "__main__":
    demo.launch()
