import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Runtime theme toggle
# ---------------------------
st.set_page_config(page_title="bold-binders · Phage Infection Classifiers · DASHBOARD", layout="wide")
mode = st.toggle("Dark mode", value=True)

THEME = {
    "dark": {
        "mpl_style": "dark_background",
        "sns_style": "darkgrid",
        "bg_page": "#0A0A0A",
        "bg_fig": "#0E0E0E",
        "bg_ax": "#111111",
        "text_primary": "#FFFFFF",
        "text_secondary": "#EAEAEA",
        "text_muted": "#AAAAAA",
        "grid_alpha": 0.7,
        "heatmap_annot": "#FFFFFF",
        "bar_edge": "none",
        "palette": {
            "precision": "#00BFFF",
            "recall": "#FF7F50",
            "f1": "#90EE90"
        },
        "heatmap_cmap": "viridis",
        "linecolor": "#2A2A2A",
    },
    "light": {
        "mpl_style": "default",
        "sns_style": "whitegrid",
        "bg_page": "#FFFFFF",
        "bg_fig": "#FFFFFF",
        "bg_ax": "#FAFAFA",
        "text_primary": "#111111",
        "text_secondary": "#222222",
        "text_muted": "#555555",
        "grid_alpha": 0.4,
        "heatmap_annot": "#000000",
        "bar_edge": "none",
        "palette": {
            # light-friendly defaults
            "precision": "#1f77b4",
            "recall": "#ff7f0e",
            "f1": "#2ca02c"
        },
        "heatmap_cmap": "viridis",
        "linecolor": "#DDDDDD",
    },
}

theme = THEME["dark" if mode else "light"]

# Apply plotting themes
plt.style.use(theme["mpl_style"])
sns.set_theme(style=theme["sns_style"])

# ---------------------------
# Precomputed Metrics (unchanged)
# ---------------------------
metrics = {
    "Decision Tree": {
        "accuracy": 0.39,
        "precision": {"early": 0.44, "middle": 0.24, "late": 0.41},
        "recall": {"early": 0.49, "middle": 0.19, "late": 0.42},
        "f1": {"early": 0.46, "middle": 0.21, "late": 0.42},
        "confusion_matrix": [
            [51, 20, 33],
            [35, 12, 17],
            [30, 18, 35]
        ]
    },
    "Random Forest": {
        "accuracy": 0.48,
        "precision": {"early": 0.47, "middle": 0.50, "late": 0.53},
        "recall": {"early": 0.87, "middle": 0.03, "late": 0.35},
        "f1": {"early": 0.61, "middle": 0.06, "late": 0.42},
        "confusion_matrix": [
            [90, 1, 13],
            [49, 2, 13],
            [53, 1, 29]
        ]
    },
    "Support Vector Machine": {
        "accuracy": 0.49,
        "precision": {"early": 0.56, "middle": 0.50, "late": 0.41},
        "recall": {"early": 0.48, "middle": 0.67, "late": 0.37},
        "f1": {"early": 0.52, "middle": 0.57, "late": 0.39},
        "confusion_matrix": [
            [50, 20, 34],
            [10, 43, 11],
            [29, 23, 31]
        ]
    },
    "Multi Layer Perceptron": {
        "accuracy": 0.53,
        "precision": {"early": 0.59, "middle": 0.45, "late": 0.51},
        "recall": {"early": 0.56, "middle": 0.37, "late": 0.62},
        "f1": {"early": 0.58, "middle": 0.40, "late": 0.56},
        "confusion_matrix": [
            [73, 23, 34],
            [25, 29, 25],
            [25, 13, 62]
        ]
    }
}
labels = ["early", "middle", "late"]

# ---------------------------
# Helper for styled headers
# ---------------------------
def banner_html():
    return f"""
    <div style="text-align:center; line-height:1.05;">
        <h1 style="margin:0; color:{theme['text_primary']}; font-weight:800; font-size:3.5em;">bold-binders</h1>
        <h3 style="margin:0; color:{theme['text_secondary']}; font-weight:600; font-size:1.8em;">Phage Infection Classifiers</h3>
        <h4 style="margin:6px 0 8px 0; color:{theme['text_muted']}; letter-spacing:3px; font-size:1.4em;">DASHBOARD</h4>
    </div>
    """

def model_title_html(model_choice):
    return f"<h2 style='text-align:center; color:{theme['text_primary']}; font-weight:700; margin-top:6px;'>{model_choice}</h2>"

# ---------------------------
# Plotting Helpers (colorized)
# ---------------------------
def plot_metrics_grid(model_name):
    data = metrics[model_name]
    classes = ["early", "middle", "late"]

    macro_p = sum(data["precision"][c] for c in classes) / 3.0
    macro_r = sum(data["recall"][c] for c in classes) / 3.0
    macro_f = sum(data["f1"][c] for c in classes) / 3.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor(theme["bg_fig"])

    def base_style(ax, title=None, ylabel=None):
        ax.set_facecolor(theme["bg_ax"])
        ax.tick_params(colors=theme["text_secondary"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if title is not None:
            ax.set_title(title, color=theme["text_primary"], fontsize=13, pad=8, fontweight="bold")
        if ylabel is not None:
            ax.set_ylabel(ylabel, color=theme["text_secondary"])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=theme["grid_alpha"])
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)

    def draw_per_class(ax, values_dict, color, title, show_ylabel=False):
        base_style(ax, title=title, ylabel="Value" if show_ylabel else None)
        vals = [values_dict[c] for c in classes]
        bars = ax.bar(classes, vals, color=color, edgecolor=theme["bar_edge"])
        ax.set_ylim(0, 1)
        ax.set_xticklabels(classes, rotation=0, color=theme["text_secondary"])
        ax.yaxis.set_tick_params(colors=theme["text_secondary"])
        ax.bar_label(bars, fmt="%.2f", label_type="edge", padding=3, fontsize=9, color=theme["text_secondary"])

    draw_per_class(axes[0, 0], data["precision"], theme["palette"]["precision"], "Precision", show_ylabel=False)
    draw_per_class(axes[0, 1], data["recall"], theme["palette"]["recall"], "Recall", show_ylabel=False)
    draw_per_class(axes[1, 0], data["f1"], theme["palette"]["f1"], "F1", show_ylabel=False)

    ax = axes[1, 1]
    base_style(ax, title="Macro Averages", ylabel=None)
    labels_macro = ["Precision", "Recall", "F1"]
    values_macro = [macro_p, macro_r, macro_f]
    colors_macro = [theme["palette"]["precision"], theme["palette"]["recall"], theme["palette"]["f1"]]
    bars = ax.bar(labels_macro, values_macro, color=colors_macro, edgecolor=theme["bar_edge"])
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels_macro, rotation=0, color=theme["text_secondary"])
    ax.yaxis.set_tick_params(colors=theme["text_secondary"])
    ax.bar_label(bars, fmt="%.2f", label_type="edge", padding=3, fontsize=9, color=theme["text_secondary"])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(theme["bg_fig"])
    ax.set_facecolor(theme["bg_ax"])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=theme["heatmap_cmap"],
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor=theme["linecolor"],
        annot_kws={"color": theme["heatmap_annot"], "fontsize": 11},
        ax=ax,
    )

    ax.set_xlabel("Predicted", color=theme["text_secondary"])
    ax.set_ylabel("Actual", color=theme["text_secondary"])
    ax.tick_params(colors=theme["text_secondary"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_test_sample(cm):
    actual_counts = [sum(row) for row in cm]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(theme["bg_fig"])
    ax.set_facecolor(theme["bg_ax"])

    bars = ax.barh(labels, actual_counts, edgecolor=theme["bar_edge"], color=theme["palette"]["precision"])
    ax.invert_yaxis()
    ax.set_xticklabels(ax.get_xticks(), color=theme["text_secondary"])
    ax.set_yticklabels(labels, color=theme["text_secondary"])
    ax.tick_params(colors=theme["text_secondary"])

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=theme["grid_alpha"])
    ax.grid(False, axis="y")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9, color=theme["text_secondary"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Test Sample", fontsize=14, color=theme["text_primary"], fontweight="bold")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_sample_and_confusion(cm):
    actual_counts = [sum(row) for row in cm]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor(theme["bg_fig"])

    ax0 = axes[0]
    ax0.set_facecolor(theme["bg_ax"])
    bars = ax0.barh(labels, actual_counts, edgecolor=theme["bar_edge"], color=theme["palette"]["precision"])
    ax0.invert_yaxis()
    ax0.tick_params(colors=theme["text_secondary"])
    ax0.set_xlabel("")
    ax0.set_ylabel("")
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=theme["grid_alpha"])
    ax0.grid(False, axis="y")
    ax0.bar_label(bars, fmt="%d", padding=3, fontsize=9, color=theme["text_secondary"])
    ax0.set_title("Test Sample", fontsize=13, color=theme["text_primary"], fontweight="bold")

    ax1 = axes[1]
    ax1.set_facecolor(theme["bg_ax"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=theme["heatmap_cmap"],
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor=theme["linecolor"],
        annot_kws={"color": theme["heatmap_annot"], "fontsize": 11},
        ax=ax1,
    )
    ax1.set_xlabel("Predicted", color=theme["text_secondary"], fontweight="bold")
    ax1.set_ylabel("Actual", color=theme["text_secondary"], fontweight="bold")
    ax1.tick_params(colors=theme["text_secondary"])
    ax1.set_title("Confusion Matrix", fontsize=13, color=theme["text_primary"], fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.grid(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_macro_comparison():
    models = list(metrics.keys())
    precisions, recalls, f1s = [], [], []

    for model in models:
        data = metrics[model]
        classes_local = ["early", "middle", "late"]
        precisions.append(sum(data["precision"][c] for c in classes_local) / 3)
        recalls.append(sum(data["recall"][c] for c in classes_local) / 3)
        f1s.append(sum(data["f1"][c] for c in classes_local) / 3)

    x = list(range(len(models)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(theme["bg_fig"])
    ax.set_facecolor(theme["bg_ax"])

    bars_p = ax.bar([i - width for i in x], precisions, width=width, color=theme["palette"]["precision"], edgecolor=theme["bar_edge"])
    bars_r = ax.bar(x,                         recalls,   width=width, color=theme["palette"]["recall"],    edgecolor=theme["bar_edge"])
    bars_f = ax.bar([i + width for i in x],   f1s,        width=width, color=theme["palette"]["f1"],       edgecolor=theme["bar_edge"])

    ax.bar_label(bars_p, labels=[f"{v:.2f}" for v in precisions], padding=3, fontsize=9, color=theme["text_secondary"])
    ax.bar_label(bars_r, labels=[f"{v:.2f}" for v in recalls],   padding=3, fontsize=9, color=theme["text_secondary"])
    ax.bar_label(bars_f, labels=[f"{v:.2f}" for v in f1s],       padding=3, fontsize=9, color=theme["text_secondary"])

    ax.set_xticks(x)
    ax.set_xticklabels(models, color=theme["text_secondary"], rotation=0, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(colors=theme["text_secondary"])
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=theme["grid_alpha"])
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------
# Streamlit UI (colors wired to theme)
# ---------------------------
# Page background & body text color
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {theme['bg_page']};
        color: {theme['text_secondary']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Column widths: first, second empty, middle wider, fourth empty, last
c1, c2, c3, c4, c5 = st.columns([1.1, 0.6, 1.6, 0.6, 1.1])

with c1:
    st.image("./.media/tp25_avatar2.png", use_container_width=True)

# c2 left intentionally empty

with c3:
    st.markdown(banner_html(), unsafe_allow_html=True)  # banner once (above)
    model_options = list(metrics.keys()) + ["Comparison"]
    model_choice = st.selectbox(" ", model_options, label_visibility="collapsed", key="model_select_center")
    st.markdown(model_title_html(model_choice), unsafe_allow_html=True)  # title once (below)

# c4 left intentionally empty

with c5:
    st.image("./.media/tp25_avatar1.png", use_container_width=True)

# Content below header (unchanged)
if model_choice == "Comparison":
    plot_macro_comparison()
else:
    acc = metrics[model_choice]["accuracy"]
    st.subheader("Accuracy")
    st.markdown(
        f"<h3 style='text-align: left; color: {theme['text_primary']};'>{int(acc * 100)}%</h3>",
        unsafe_allow_html=True,
    )

    st.subheader("Metrics")
    plot_metrics_grid(model_choice)

    st.subheader("Sample & Confusion")
    plot_sample_and_confusion(metrics[model_choice]["confusion_matrix"])
