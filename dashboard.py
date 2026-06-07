"""Real-time Muse S dashboard built with Plotly Dash."""

import numpy as np
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from config import (
    EEG_CHANNELS, EEG_SAMPLE_RATE, EEG_DISPLAY_SECONDS,
    PPG_CHANNELS, PPG_SAMPLE_RATE,
    ACC_CHANNELS, GYRO_CHANNELS, IMU_SAMPLE_RATE,
    FREQ_BANDS, ELECTRODE_POSITIONS,
    DASH_UPDATE_INTERVAL_MS,
)
from data_store import MuseDataStore
from signal_processing import (
    filter_eeg_channel, compute_all_band_powers,
    compute_signal_quality, compute_heart_rate, compute_psd,
)

BAND_COLORS = {
    "Delta": "#636EFA",
    "Theta": "#EF553B",
    "Alpha": "#00CC96",
    "Beta":  "#AB63FA",
    "Gamma": "#FFA15A",
}

CH_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

BG = "#0f0f1a"
CARD_BG = "#1a1a2e"
GRID = "#2a2a4a"


def _layout(**kw):
    base = dict(
        paper_bgcolor=CARD_BG,
        plot_bgcolor="#111128",
        font=dict(color="#ccc", family="monospace", size=11),
    )
    if "margin" not in kw:
        base["margin"] = dict(l=50, r=15, t=10, b=35)
    base.update(kw)
    return base


def _axis():
    return dict(gridcolor=GRID, zerolinecolor=GRID)


def create_app(store: MuseDataStore) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Muse S | Neural Dashboard",
        update_title=None,
    )

    card_style = {"backgroundColor": CARD_BG, "border": "1px solid #333", "borderRadius": "8px"}

    app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh", "padding": "15px"}, children=[
        html.H2("MUSE S  |  NEURAL DASHBOARD",
                 style={"fontWeight": "700", "letterSpacing": "3px", "color": "#00CC96",
                        "marginBottom": "4px", "fontFamily": "monospace"}),
        html.Div(id="status-bar", style={"fontSize": "0.85rem", "color": "#888", "marginBottom": "15px"}),

        # Row 1: EEG + Band Power
        dbc.Row([
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("RAW EEG", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                dcc.Graph(id="eeg-traces", style={"height": "330px"}, config={"displayModeBar": False}),
            ]), width=8),
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("BAND POWER", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                dcc.Graph(id="band-power", style={"height": "330px"}, config={"displayModeBar": False}),
            ]), width=4),
        ], className="mb-3"),

        # Row 2: PSD + Topo + Quality
        dbc.Row([
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("POWER SPECTRAL DENSITY", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                dcc.Graph(id="psd-plot", style={"height": "280px"}, config={"displayModeBar": False}),
            ]), width=5),
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("TOPOGRAPHIC MAP", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                dcc.Graph(id="topo-map", style={"height": "280px"}, config={"displayModeBar": False}),
            ]), width=4),
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("SIGNAL QUALITY", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                html.Div(id="quality-bars", style={"padding": "12px"}),
            ]), width=3),
        ], className="mb-3"),

        # Row 3: PPG + ACC + GYRO
        dbc.Row([
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("PPG / HEART RATE", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                dcc.Graph(id="ppg-plot", style={"height": "240px"}, config={"displayModeBar": False}),
            ]), width=4),
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("ACCELEROMETER", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                dcc.Graph(id="acc-plot", style={"height": "240px"}, config={"displayModeBar": False}),
            ]), width=4),
            dbc.Col(html.Div(style=card_style, children=[
                html.Div("GYROSCOPE", style={"color": "#666", "padding": "8px 12px", "fontSize": "0.75rem"}),
                dcc.Graph(id="gyro-plot", style={"height": "240px"}, config={"displayModeBar": False}),
            ]), width=4),
        ]),

        dcc.Interval(id="tick", interval=DASH_UPDATE_INTERVAL_MS, n_intervals=0),
    ])

    @app.callback(
        Output("status-bar", "children"),
        Output("eeg-traces", "figure"),
        Output("band-power", "figure"),
        Output("psd-plot", "figure"),
        Output("topo-map", "figure"),
        Output("quality-bars", "children"),
        Output("ppg-plot", "figure"),
        Output("acc-plot", "figure"),
        Output("gyro-plot", "figure"),
        Input("tick", "n_intervals"),
    )
    def update_all(_n):
        return (
            _status(store),
            _eeg(store),
            _bands(store),
            _psd(store),
            _topo(store),
            _quality(store),
            _ppg(store),
            _imu(store, store.acc, ACC_CHANNELS, ["#636EFA", "#EF553B", "#00CC96"]),
            _imu(store, store.gyro, GYRO_CHANNELS, ["#AB63FA", "#FFA15A", "#19D3F3"]),
        )

    return app


# ── Status ──────────────────────────────────────────────

def _status(s: MuseDataStore) -> str:
    if not s.connected:
        return "DISCONNECTED"
    total = s.eeg.total_written
    secs = total / EEG_SAMPLE_RATE
    return f"CONNECTED  |  {s.device_name}  |  {total:,} samples  |  {secs:.1f}s"


# ── EEG Traces ──────────────────────────────────────────

def _eeg(s: MuseDataStore) -> go.Figure:
    n = EEG_SAMPLE_RATE * EEG_DISPLAY_SECONDS
    data, _ = s.eeg.get_last_n(n)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    for i in range(4):
        if data.shape[1] > 10:
            t = np.arange(data.shape[1]) / EEG_SAMPLE_RATE
            y = filter_eeg_channel(data[i])
        else:
            t, y = [], []

        fig.add_trace(go.Scattergl(
            x=t, y=y, name=EEG_CHANNELS[i],
            line=dict(color=CH_COLORS[i], width=1), mode="lines",
        ), row=i+1, col=1)

        fig.update_yaxes(
            title_text=EEG_CHANNELS[i], row=i+1, col=1,
            **_axis(), title_font=dict(size=10),
        )

    fig.update_layout(
        **_layout(height=320, showlegend=False),
        margin=dict(l=60, r=10, t=5, b=30),
    )
    fig.update_xaxes(title_text="sec", row=4, col=1, **_axis())
    for i in range(1, 4):
        fig.update_xaxes(**_axis(), row=i, col=1)
    return fig


# ── Band Power ──────────────────────────────────────────

def _bands(s: MuseDataStore) -> go.Figure:
    data, _ = s.eeg.get_last_n(EEG_SAMPLE_RATE * 2)
    bands = list(FREQ_BANDS.keys())

    if data.shape[1] >= EEG_SAMPLE_RATE:
        powers = compute_all_band_powers(data)
        avg = [float(np.mean(powers[b])) for b in bands]
    else:
        avg = [0.0] * len(bands)

    fig = go.Figure(go.Bar(
        x=bands, y=avg,
        marker_color=[BAND_COLORS[b] for b in bands],
        text=[f"{v:.1f}" for v in avg], textposition="auto",
        textfont=dict(size=11, color="white"),
    ))
    fig.update_layout(**_layout(height=320, showlegend=False), yaxis=dict(title="uV²/Hz", **_axis()), xaxis=_axis())
    return fig


# ── PSD ─────────────────────────────────────────────────

def _psd(s: MuseDataStore) -> go.Figure:
    data, _ = s.eeg.get_last_n(EEG_SAMPLE_RATE * 2)
    fig = go.Figure()

    if data.shape[1] >= EEG_SAMPLE_RATE:
        for i in range(4):
            filtered = filter_eeg_channel(data[i])
            power, freqs = compute_psd(filtered)
            if len(freqs) > 0:
                mask = freqs <= 50
                fig.add_trace(go.Scattergl(
                    x=freqs[mask], y=power[mask], name=EEG_CHANNELS[i],
                    line=dict(color=CH_COLORS[i], width=1.5), mode="lines",
                ))
        for name, (lo, hi) in FREQ_BANDS.items():
            fig.add_vrect(x0=lo, x1=hi, fillcolor=BAND_COLORS[name], opacity=0.08, layer="below", line_width=0)

    fig.update_layout(
        **_layout(height=270),
        xaxis=dict(title="Hz", **_axis()), yaxis=dict(title="Power", type="log", **_axis()),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
    )
    return fig


# ── Topo Map ────────────────────────────────────────────

def _topo(s: MuseDataStore) -> go.Figure:
    data, _ = s.eeg.get_last_n(EEG_SAMPLE_RATE * 2)
    fig = go.Figure()

    alpha = np.zeros(4)
    if data.shape[1] >= EEG_SAMPLE_RATE:
        powers = compute_all_band_powers(data)
        for i in range(4):
            total = sum(powers[b][i] for b in FREQ_BANDS)
            alpha[i] = powers["Alpha"][i] / total if total > 0 else 0

    xs = [ELECTRODE_POSITIONS[ch][0] for ch in EEG_CHANNELS]
    ys = [ELECTRODE_POSITIONS[ch][1] for ch in EEG_CHANNELS]

    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scattergl(x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
                                mode="lines", line=dict(color="#555", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scattergl(x=[-.08, 0, .08], y=[.98, 1.12, .98],
                                mode="lines", line=dict(color="#555", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=40, color=alpha.tolist(), colorscale="Viridis", cmin=0, cmax=0.5,
                    colorbar=dict(title="Alpha", len=0.8, thickness=12), line=dict(width=2, color="#ddd")),
        text=EEG_CHANNELS, textposition="middle center", textfont=dict(size=9, color="white"),
    ))

    fig.update_layout(
        **_layout(height=270, showlegend=False),
        xaxis=dict(range=[-1.4, 1.4], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-1.3, 1.3], visible=False),
    )
    return fig


# ── Signal Quality ──────────────────────────────────────

def _quality(s: MuseDataStore) -> list:
    data, _ = s.eeg.get_last_n(EEG_SAMPLE_RATE * 2)
    q = compute_signal_quality(data) if data.shape[1] >= EEG_SAMPLE_RATE else np.zeros(4)

    bars = []
    for i, ch in enumerate(EEG_CHANNELS):
        v = q[i]
        color = "#00CC96" if v >= 0.6 else ("#FFA15A" if v >= 0.3 else "#EF553B")
        label = "GOOD" if v >= 0.6 else ("FAIR" if v >= 0.3 else "POOR")
        bars.append(html.Div([
            html.Div([
                html.Span(ch, style={"fontWeight": "600", "color": "#ccc"}),
                html.Span(f" {label}", style={"color": color, "fontSize": "0.75rem", "marginLeft": "8px"}),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
            html.Div(style={"height": "8px", "backgroundColor": "#222", "borderRadius": "4px", "overflow": "hidden"}, children=[
                html.Div(style={"height": "100%", "width": f"{v*100:.0f}%", "backgroundColor": color, "borderRadius": "4px"}),
            ]),
        ], style={"marginBottom": "16px"}))
    return bars


# ── PPG ─────────────────────────────────────────────────

def _ppg(s: MuseDataStore) -> go.Figure:
    data, _ = s.ppg.get_last_n(PPG_SAMPLE_RATE * 10)
    fig = go.Figure()

    hr = compute_heart_rate(data) if data.shape[1] > PPG_SAMPLE_RATE * 5 else 0.0

    if data.shape[1] > 10:
        t = np.arange(data.shape[1]) / PPG_SAMPLE_RATE
        fig.add_trace(go.Scattergl(x=t, y=data[1], name="IR",
                                    line=dict(color="#EF553B", width=1.5), mode="lines"))

    hr_text = f"{hr:.0f} BPM" if hr > 0 else "-- BPM"
    fig.update_layout(
        **_layout(height=230, showlegend=False),
        xaxis=dict(title="sec", **_axis()), yaxis=_axis(),
        annotations=[dict(text=hr_text, x=0.98, y=0.92, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=22, color="#EF553B", family="monospace"))],
    )
    return fig


# ── IMU (shared for ACC/GYRO) ───────────────────────────

def _imu(s: MuseDataStore, buf, channels, colors) -> go.Figure:
    data, _ = buf.get_last_n(IMU_SAMPLE_RATE * 5)
    fig = go.Figure()

    if data.shape[1] > 10:
        t = np.arange(data.shape[1]) / IMU_SAMPLE_RATE
        for i, name in enumerate(channels):
            fig.add_trace(go.Scattergl(x=t, y=data[i], name=name,
                                        line=dict(color=colors[i], width=1.2), mode="lines"))

    fig.update_layout(
        **_layout(height=230),
        xaxis=dict(title="sec", **_axis()), yaxis=_axis(),
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=9)),
    )
    return fig
