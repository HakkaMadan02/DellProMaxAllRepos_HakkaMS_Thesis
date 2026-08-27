import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

out = Path("output")
out.mkdir(exist_ok=True)

COLORS = {
    "Accuracy": "#7A0019",   # blood red
    "Precision": "#111111",  # jet black
    "Recall": "#8C8C8C",     # grey
}

DATA = pd.DataFrame({
    "MLP": ["Message", "Aggregate", "Update", "CBF-Head"],
    "Accuracy": [99.92, 99.86, 92.23, 91.49],
    "Precision": [99.94, 100.00, 94.40, 95.15],
    "Recall": [99.88, 99.72, 89.87, 87.46],
})


def save_metadata(path: Path, caption: str, description: str):
    with open(path.with_suffix(path.suffix + ".meta.json"), "w") as f:
        json.dump(
            {"caption": caption, "description": description},
            f,
            indent=2
        )


def make_bar_chart(df: pd.DataFrame, out_path: Path):
    fig = go.Figure()
    metrics = ["Accuracy", "Precision", "Recall"]

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=df["MLP"],
            y=df[metric],
            name=metric,
            marker_color=COLORS[metric],
            offsetgroup=str(i),
            hovertemplate=f"{metric}<br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=18, family="DejaVu Sans"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=18, family="DejaVu Sans", color="black"),
            traceorder="normal",
        ),
        margin=dict(t=120, r=35, b=80, l=90),
    )

    fig.update_xaxes(
        title_text="MLP Stage",
        title_font=dict(size=22, color="black", family="DejaVu Sans"),
        tickfont=dict(size=20, color="black", family="DejaVu Sans"),
        showline=True,
        linewidth=1.8,
        linecolor="black",
        showgrid=False,
        ticks="outside",
    )

    fig.update_yaxes(
        title_text="Metric (%)",
        title_font=dict(size=22, color="black", family="DejaVu Sans"),
        tickfont=dict(size=20, color="black", family="DejaVu Sans"),
        range=[85, 100],
        tickmode="array",
        tickvals=[85, 90, 95, 100],
        ticksuffix="%",
        showline=True,
        linewidth=1.8,
        linecolor="black",
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
    )

    fig.write_image(str(out_path))
    save_metadata(
        out_path,
        "Final grouped bar chart for GCBF+ probe metrics",
        "Grouped bar chart for Message, Aggregate, Update, and CBF-Head using test accuracy, precision, and recall.",
    )


def make_slope_chart(out_path: Path):
    x = ["Aggregate", "Update"]
    values = {
        "Accuracy": [99.86, 92.23],
        "Precision": [100.00, 94.40],
        "Recall": [99.72, 89.87],
    }

    fig = go.Figure()

    for metric in ["Accuracy", "Precision", "Recall"]:
        fig.add_trace(go.Scatter(
            x=x,
            y=values[metric],
            mode="lines+markers",
            showlegend=False,
            line=dict(color=COLORS[metric], width=5),
            marker=dict(size=12, color=COLORS[metric]),
            hovertemplate=f"{metric}<br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
        ))

    for xpos in x:
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=xpos,
            x1=xpos,
            y0=88,
            y1=100,
            line=dict(color="black", width=3),
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=18, family="DejaVu Sans"),
        showlegend=False,
        margin=dict(t=40, r=40, b=80, l=90),
    )

    fig.update_xaxes(
        title_text="Metric Drop Rate",
        title_font=dict(size=22, color="black", family="DejaVu Sans"),
        tickfont=dict(size=20, color="black", family="DejaVu Sans"),
        showline=False,
        showgrid=False,
    )

    fig.update_yaxes(
        title_text="Metric (%)",
        title_font=dict(size=22, color="black", family="DejaVu Sans"),
        tickfont=dict(size=20, color="black", family="DejaVu Sans"),
        range=[88, 100],
        tickmode="array",
        tickvals=[88, 90, 92, 94, 96, 98, 100],
        ticksuffix="%",
        showline=True,
        linewidth=1.8,
        linecolor="black",
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
    )

    fig.write_image(str(out_path))
    save_metadata(
        out_path,
        "Final slope chart for Aggregate to Update",
        "Minimal two-point slope chart showing the drop in accuracy, precision, and recall from Aggregate to Update.",
    )


if __name__ == "__main__":
    make_bar_chart(DATA, out / "final_bar_chart.png")
    make_slope_chart(out / "final_line_chart.png")
    print("Saved:")
    print(" - output/final_bar_chart.png")
    print(" - output/final_line_chart.png")