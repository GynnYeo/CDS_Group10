from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.inference import MultiTaskInferenceRunner
from gui.map_utils import build_prediction_map_html

DEFAULT_DATASET = "earthquake_aftershock_v2_gcmt"
DEFAULT_RUN = "mlp_pm_bce_countw05_enriched_256_128_64_d03_lr3e4_alltasks"
BACKGROUND_PATH = PROJECT_ROOT / "gui" / "assets" / "background.jpg"

_RUNNER_CACHE: dict[tuple[str, str], MultiTaskInferenceRunner] = {}


def get_runner(dataset_name: str, run_name: str) -> MultiTaskInferenceRunner:
    key = (dataset_name, run_name)
    if key not in _RUNNER_CACHE:
        _RUNNER_CACHE[key] = MultiTaskInferenceRunner(dataset_name, run_name)
    return _RUNNER_CACHE[key]


def load_uploaded_json(file_path: str | None):
    if file_path is None:
        return None, "", "", gr.update(visible=False)

    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    pretty_json = json.dumps(payload, indent=2)
    details_html = format_mainquake_details(payload)
    return payload, pretty_json, details_html, gr.update(visible=True)


def format_mainquake_details(payload: dict) -> str:
    event_id = payload.get("trigger_event_id", "Unknown")
    magnitude = payload.get("magnitude", payload.get("trigger_magnitude", "N/A"))
    latitude = payload.get("latitude", payload.get("trigger_latitude", "N/A"))
    longitude = payload.get("longitude", payload.get("trigger_longitude", "N/A"))
    depth_km = payload.get("depth_km", payload.get("trigger_depth_km", "N/A"))
    trigger_time = payload.get("trigger_time", "N/A")

    return f"""
    <div class="glass-card">
      <div class="section-title">Mainquake Details</div>
      <div class="mainquake-grid">
        <div class="info-card">
          <div class="info-label">Event ID</div>
          <div class="info-value">{event_id}</div>
        </div>
        <div class="info-card">
          <div class="info-label">Magnitude</div>
          <div class="info-value">{magnitude}</div>
        </div>
        <div class="info-card">
          <div class="info-label">Depth (km)</div>
          <div class="info-value">{depth_km}</div>
        </div>
        <div class="info-card">
          <div class="info-label">Latitude</div>
          <div class="info-value">{latitude}</div>
        </div>
        <div class="info-card">
          <div class="info-label">Longitude</div>
          <div class="info-value">{longitude}</div>
        </div>
        <div class="info-card">
          <div class="info-label">Time</div>
          <div class="info-value">{trigger_time}</div>
        </div>
      </div>
    </div>
    """


def format_prediction_summary(result: dict) -> str:
    return f"""
    <div class="prediction-grid">
      <div class="glass-card">
        <div class="section-title">Task 1 — Aftershock Probability</div>
        <div class="metric-big">24h: {result['prob_24h']:.3f}</div>
        <div class="metric-big">72h: {result['prob_72h']:.3f}</div>
      </div>
      <div class="glass-card">
        <div class="section-title">Task 2 — Predicted Count</div>
        <div class="metric-big">24h: {max(0, round(result['count_24h']))}</div>
        <div class="metric-big">72h: {max(0, round(result['count_72h']))}</div>
      </div>
      <div class="glass-card">
        <div class="section-title">Task 3 — Predicted Max Magnitude</div>
        <div class="metric-big">24h: {result['mag_24h']:.2f}</div>
        <div class="metric-big">72h: {result['mag_72h']:.2f}</div>
      </div>
    </div>
    """


def predict_from_upload(file_path, dataset_name, run_name):
    if file_path is None:
        raise gr.Error("Please upload a JSON file.")

    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    runner = get_runner(dataset_name, run_name)
    result = runner.predict_one(payload)

    df = pd.DataFrame([result])
    map_html = build_prediction_map_html(df)
    summary_html = format_prediction_summary(result)
    pretty_json = json.dumps(payload, indent=2)

    return summary_html, df, pretty_json, map_html


def build_css() -> str:
    bg_css = ""
    if BACKGROUND_PATH.exists():
        with open(BACKGROUND_PATH, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        bg_css = f"""
        .gradio-container {{
            background-image: linear-gradient(rgba(0,0,0,0.58), rgba(0,0,0,0.58)), url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg_css = """
        .gradio-container {
            background: linear-gradient(135deg, #0b1610, #13231a);
        }
        """

    return f"""
    {bg_css}

    .block-title {{
        text-align: center;
        margin-bottom: 20px;
    }}

    .block-title h1 {{
        font-size: 46px;
        color: white;
        margin-bottom: 6px;
        font-weight: 800;
    }}

    .block-title p {{
        font-size: 18px;
        color: #e6f0e9;
        margin-top: 0;
    }}

    .glass-card {{
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 18px;
        padding: 18px;
        color: white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    }}

    .section-title {{
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #f5fff7;
    }}

    .prediction-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 12px;
    }}

    .mainquake-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }}

    .info-card {{
        background: rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 12px;
    }}

    .info-label {{
        font-size: 13px;
        color: #d8eadc;
        margin-bottom: 4px;
    }}

    .info-value {{
        font-size: 18px;
        font-weight: 700;
        color: white;
        word-break: break-word;
    }}

    .metric-big {{
        font-size: 28px;
        font-weight: 800;
        margin: 8px 0;
        color: #ffffff;
    }}

    .panel-label {{
        color: white;
        font-weight: 700;
        margin-bottom: 8px;
    }}

    @media (max-width: 900px) {{
        .prediction-grid,
        .mainquake-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    """


with gr.Blocks(css=build_css(), title="Earthquake Aftershock Prediction System") as demo:
    gr.Markdown(
        """
        <div class="block-title">
          <h1>🌍 Earthquake Aftershock Prediction System</h1>
          <p>Upload a mainquake JSON, inspect its details, and predict Tasks 1–3 on the spot.</p>
        </div>
        """
    )

    uploaded_payload = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload JSON", file_types=[".json"])
            dataset_box = gr.Textbox(value=DEFAULT_DATASET, label="Dataset")
            # run_box = gr.Textbox(value=DEFAULT_RUN, label="Run name")
            run_box = gr.Dropdown(
            choices=[
                "mlp_pm_bce_countw05_enriched_256_128_64_d03_lr3e4_alltasks",
                "mlp_count_twohead_params",
                "mlp_count_bce_countw10_enriched_256_128_64_d03_lr3e4",
            ],
            value=DEFAULT_RUN,
            label="Run name"
            )
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=2):
            mainquake_details = gr.HTML(visible=False)

    prediction_summary = gr.HTML()

    prediction_df = gr.Dataframe(label="Predictions", interactive=False)

    map_html = gr.HTML(label="Map")

    uploaded_json = gr.Code(label="Uploaded JSON", language="json")

    file_input.change(
        fn=load_uploaded_json,
        inputs=[file_input],
        outputs=[uploaded_payload, uploaded_json, mainquake_details, mainquake_details],
    )

    predict_btn.click(
        fn=predict_from_upload,
        inputs=[file_input, dataset_box, run_box],
        outputs=[prediction_summary, prediction_df, uploaded_json, map_html],
    )


if __name__ == "__main__":
    demo.launch()