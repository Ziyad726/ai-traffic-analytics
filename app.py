
import streamlit as st
import pandas as pd
from pathlib import Path

# =====================================
# 1. Application Configuration
# =====================================

st.set_page_config(
    page_title="AI Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# =====================================
# 2. Dashboard Styling
# =====================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background-color: #17243A;
    border: 1px solid #29425A;
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 3. Load Data
# =====================================

@st.cache_data
def load_data():
    frames = pd.read_csv(
        DATA / "traffic_frame_metrics.csv"
    )
    classes = pd.read_csv(
        DATA / "traffic_class_distribution.csv"
    )
    return frames, classes

df, classes = load_data()

# =====================================
# 4. Header
# =====================================

st.title("🚦 AI Traffic Analytics")

st.markdown(
    "**Computer Vision-Based Traffic Monitoring**"
)

st.caption(
    "YOLO11s Fine-Tuned + ByteTrack | "
    "EMT Dataset | Research Prototype"
)

st.info(
    "This dashboard shows 100 consecutive frames "
    "from one EMT training video. Detection and "
    "tracking outputs are experimental. "
    "Bounding-box observations are not unique "
    "vehicle counts."
)

# =====================================
# 5. Sidebar Controls
# =====================================

st.sidebar.title("⚙️ Dashboard Controls")

start_frame, end_frame = st.sidebar.slider(
    "Frame Range",
    min_value=1,
    max_value=100,
    value=(1, 100)
)

filtered = df[
    df["frame_id"].between(
        start_frame, end_frame
    )
].copy()

series_options = {
    "Ground Truth Cars": "GT_Cars",
    "YOLO Detections": "YOLO_Detections",
    "ByteTrack Outputs": "ByteTrack_Outputs",
    "All Annotated Objects": "GT_All_Objects"
}

selected_series = st.sidebar.multiselect(
    "Show on Traffic Chart",
    options=list(series_options.keys()),
    default=[
        "Ground Truth Cars",
        "YOLO Detections",
        "ByteTrack Outputs"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("**Experiment Settings**")
st.sidebar.write("Model: Fine-Tuned YOLO11s")
st.sidebar.write("Tracking: ByteTrack Default")
st.sidebar.write("Input resolution: 1280")
st.sidebar.write("Detection confidence: 0.10")
st.sidebar.write("Video: video_054604")

# =====================================
# 6. KPI Overview
# =====================================

st.subheader("Traffic Overview")


c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Frames Selected",
        len(filtered)
    )

with c2:
    st.metric(
        "GT Car Observations",
        f"{filtered['GT_Cars'].sum():,}"
    )

c3, c4 = st.columns(2)

with c3:
    st.metric(
        "YOLO Detection Boxes",
        f"{filtered['YOLO_Detections'].sum():,}"
    )

with c4:
    st.metric(
        "ByteTrack Output Boxes",
        f"{filtered['ByteTrack_Outputs'].sum():,}"
    )

# =====================================
# 7. Dashboard Tabs
# =====================================

tab1, tab2, tab3 = st.tabs([
    "📊 Traffic",
    "🎯 Evaluation",
    "📋 Data"
])

# =====================================
# TAB 1: Traffic Analytics
# =====================================

with tab1:

    st.subheader("Observed Traffic Across Frames")

    st.caption(
        "Per-frame box counts. YOLO and ByteTrack "
        "may include false positives."
    )

    if selected_series:

        plot_data = filtered.rename(
            columns={
                value: key
                for key, value
                in series_options.items()
            }
        )

        st.line_chart(
            plot_data,
            x="frame_id",
            y=selected_series
        )

    else:
        st.warning(
            "Select at least one chart series."
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Ground Truth Class Distribution")

        st.caption(
            "Full 100-frame dataset; this chart "
            "does not change with the frame filter."
        )

        distribution = classes.sort_values(
            "Annotations",
            ascending=True
        )

        st.bar_chart(
            distribution,
            x="Class",
            y="Annotations",
            horizontal=True
        )

    with right:

        st.subheader("Traffic Statistics")

        st.metric(
            "Average GT Cars per Frame",
            f"{filtered['GT_Cars'].mean():.2f}"
        )

        st.metric(
            "Maximum GT Cars in a Frame",
            int(filtered["GT_Cars"].max())
        )

        st.metric(
            "Average YOLO Boxes per Frame",
            f"{filtered['YOLO_Detections'].mean():.2f}"
        )

        st.metric(
            "Average Tracked Boxes per Frame",
            f"{filtered['ByteTrack_Outputs'].mean():.2f}"
        )

    st.divider()

    preview_path = DATA / "frame100_1280.jpg"

    if preview_path.exists():

        st.subheader("Tracking Visualization")

        st.image(
            str(preview_path),
            caption=(
                "Frame 100 — Fine-Tuned YOLO11s "
                "+ Default ByteTrack at imgsz=1280"
            )
        )

# =====================================
# TAB 2: Model Evaluation
# =====================================

with tab2:

    st.subheader("Object Detection Performance")

    gt_total = int(filtered["GT_Cars"].sum())
    tp_total = int(filtered["YOLO_TP"].sum())

    recall = (
        100 * tp_total / gt_total
        if gt_total > 0 else 0
    )

    e1, e2, e3 = st.columns(3)

    e1.metric("GT Car Boxes", gt_total)
    e2.metric("Matched YOLO Boxes", tp_total)
    e3.metric("Detection Recall", f"{recall:.2f}%")

    st.caption(
        "Detection matching uses IoU >= 0.5. "
        "These are training-video diagnostics, "
        "not independent test results."
    )

    st.divider()

    validation_path = (
        DATA / "validation_image_size_comparison.csv"
    )

    if validation_path.exists():

        validation = pd.read_csv(validation_path)

        st.subheader("Validation Image Size Comparison")

        st.bar_chart(
            validation,
            x="Image_Size",
            y=[
                "Precision (%)",
                "Recall (%)",
                "F1 (%)"
            ]
        )

        st.dataframe(
            validation.round(2),
            hide_index=True
        )

        st.caption(
            "Validation metrics: 80 images from "
            "four separate validation videos."
        )

    mot_path = DATA / "mot_identity_comparison.csv"

    if mot_path.exists():

        st.divider()

        st.subheader("Multi-Object Tracking Metrics")

        mot = pd.read_csv(mot_path)

        columns = [
            "Image_Size",
            "mota",
            "idf1",
            "num_switches",
            "num_fragmentations",
            "num_misses"
        ]

        st.dataframe(
            mot[columns].round(2),
            hide_index=True
        )

        st.caption(
            "MOTA and IDF1 are from the 100-frame "
            "training-video experiment. Tracking "
            "quality has not yet been established "
            "on independent sequences."
        )


# =====================================
# V2 REVISITED TEST COMPARISON
# =====================================

with tab2:

    st.divider()

    st.subheader(
        "🚗 YOLO V1 vs V2 — Test Comparison"
    )

    st.caption(
        "Car detection on 100 sampled frames "
        "from 5 EMT test videos. "
        "This is a previously examined test set, "
        "not a fresh blind evaluation."
    )

    test_overall = pd.read_csv(
        DATA / "v1_v2_revisited_test_overall.csv"
    )

    test_per_video = pd.read_csv(
        DATA / "v1_v2_revisited_test_per_video.csv"
    )

    resolution = st.radio(
        "Input Resolution",
        options=[640, 1280],
        index=1,
        horizontal=True,
        key="v2_test_resolution"
    )

    confidence = (
        0.30 if resolution == 640 else 0.35
    )

    selected = test_overall[
        (test_overall["Image_Size"] == resolution)
        & (
            test_overall["Confidence"].round(2)
            == confidence
        )
    ].sort_values("Model")

    if len(selected) == 2:

        v1 = selected[
            selected["Model"] == "V1"
        ].iloc[0]

        v2 = selected[
            selected["Model"] == "V2"
        ].iloc[0]

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "YOLO V1 — Car F1",
                f"{v1['F1 (%)']:.2f}%"
            )

        with c2:
            st.metric(
                "YOLO V2 — Car F1",
                f"{v2['F1 (%)']:.2f}%",
                delta=(
                    f"{v2['F1 (%)'] - v1['F1 (%)']:+.2f} pp"
                )
            )

        st.caption(
            f"Image size: {resolution} | "
            f"Confidence: {confidence:.2f} | "
            "Matching IoU: 0.50"
        )

        st.subheader("F1-Score Comparison")

        st.bar_chart(
            selected,
            x="Model",
            y="F1 (%)"
        )

        st.subheader("Detection Metrics")

        st.dataframe(
            selected[
                [
                    "Model",
                    "Precision (%)",
                    "Recall (%)",
                    "F1 (%)",
                    "TP",
                    "FP",
                    "FN"
                ]
            ].round(2),
            hide_index=True
        )

        if resolution == 1280:

            st.divider()

            st.subheader(
                "Performance by Test Video"
            )

            st.bar_chart(
                test_per_video,
                x="Video",
                y=["V1_F1", "V2_F1"]
            )

            st.dataframe(
                test_per_video.round(2),
                hide_index=True
            )

            st.caption(
                "V2 improved in 3 of 5 test videos. "
                "Results declined in 2 videos."
            )

    else:
        st.error(
            "Evaluation data for this setting "
            "could not be found."
        )



# =====================================
# V2 TRACKING DIAGNOSTIC SECTION
# =====================================

with tab2:

    st.divider()

    st.subheader(
        "🚘 YOLO V1 vs V2 — Tracking"
    )

    st.caption(
        "ByteTrack Default | imgsz=1280 | "
        "conf=0.10 | 100 consecutive frames "
        "from EMT training video video_054604."
    )

    tracking_comparison = pd.read_csv(
        DATA / "v1_v2_tracking_comparison.csv"
    )

    track_by_model = (
        tracking_comparison.set_index("Model")
    )

    old = track_by_model.loc["V1"]
    new = track_by_model.loc["V2"]

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "V1 Tracking IDF1",
            f"{old['IDF1']:.2f}%"
        )

    with c2:
        st.metric(
            "V2 Tracking IDF1",
            f"{new['IDF1']:.2f}%",
            delta=(
                f"{new['IDF1'] - old['IDF1']:+.2f} pp"
            )
        )

    st.subheader("Tracking Metric Comparison")

    st.bar_chart(
        tracking_comparison,
        x="Model",
        y=["Box_F1", "MOTA", "IDF1"],
        stack=False
    )

    st.subheader("Tracking Diagnostics")

    columns = [
        "Model",
        "Tracked_Boxes",
        "Empty_Frames",
        "Recall",
        "Box_F1",
        "MOTA",
        "IDF1",
        "ID_Switches",
        "Fragments"
    ]

    st.dataframe(
        tracking_comparison[columns].round(2),
        hide_index=True
    )

    st.warning(
        "V2 increased tracking coverage, but "
        "also produced 52 ID switches and "
        "64 track fragmentations. These metrics "
        "come from a training video, not an "
        "independent MOT benchmark. Track IDs "
        "must not be treated as reliable "
        "unique-vehicle counts."
    )


# =====================================
# VALIDATION TRACKING COMPARISON 143739
# =====================================

with tab2:

    st.divider()

    st.subheader(
        "🚘 Tracking — Validation Video"
    )

    st.caption(
        "V1 vs V2 | video_143739 | "
        "100 consecutive frames | "
        "ByteTrack Default | imgsz=1280 | "
        "conf=0.10 | Matching IoU=0.50"
    )

    val_tracking = pd.read_csv(
        DATA /
        "v1_v2_validation_tracking_comparison.csv"
    )

    val_models = val_tracking.set_index("Model")

    val_v1 = val_models.loc["V1"]
    val_v2 = val_models.loc["V2"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "V1 IDF1",
            f"{val_v1['IDF1']:.2f}%"
        )

    with c2:
        st.metric(
            "V2 IDF1",
            f"{val_v2['IDF1']:.2f}%",
            delta=(
                f"{val_v2['IDF1'] - val_v1['IDF1']:+.2f} pp"
            )
        )

    with c3:
        st.metric(
            "V2 MOTA",
            f"{val_v2['MOTA']:.2f}%",
            delta=(
                f"{val_v2['MOTA'] - val_v1['MOTA']:+.2f} pp"
            )
        )

    st.subheader("Validation Tracking Metrics")

    st.bar_chart(
        val_models[
            ["Box_F1", "MOTA", "IDF1"]
        ],
        stack=False
    )

    st.dataframe(
        val_tracking[
            [
                "Model",
                "Tracked_Boxes",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "Box_F1",
                "MOTA",
                "IDF1",
                "ID_Switches",
                "Fragments"
            ]
        ].round(2),
        hide_index=True
    )

    st.info(
        "This video was excluded from model training, "
        "but belongs to the validation split used during "
        "model development. These results are not a "
        "fresh blind test. ID switches = 0 does not "
        "imply perfect tracking: V2 still misses "
        "400 of 1,209 GT Car observations."
    )


# =====================================
# NIGHT VALIDATION TRACKING 204347
# =====================================

with tab2:

    st.divider()

    st.subheader(
        "🌙 Night Validation — V1 vs V2 Tracking"
    )

    st.caption(
        "EMT video_204347 | 100 consecutive frames | "
        "696 GT Car observations | 13 GT identities | "
        "ByteTrack Default | imgsz=1280 | conf=0.10"
    )

    night = pd.read_csv(
        DATA / "v1_v2_night_tracking_comparison.csv"
    )

    night_by_model = night.set_index("Model")
    n1 = night_by_model.loc["V1"]
    n2 = night_by_model.loc["V2"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "V1 Night IDF1",
            f"{n1['IDF1']:.2f}%"
        )

    with c2:
        st.metric(
            "V2 Night IDF1",
            f"{n2['IDF1']:.2f}%",
            delta=f"{n2['IDF1'] - n1['IDF1']:+.2f} pp"
        )

    with c3:
        st.metric(
            "V2 Night MOTA",
            f"{n2['MOTA']:.2f}%",
            delta=f"{n2['MOTA'] - n1['MOTA']:+.2f} pp"
        )

    # Grouped chart, not stacked
    import altair as alt

    chart_data = night.melt(
        id_vars="Model",
        value_vars=["Box_F1", "MOTA", "IDF1"],
        var_name="Metric",
        value_name="Score"
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None),
            xOffset="Model:N",
            y=alt.Y("Score:Q", title="Score (%)"),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                "Model:N",
                "Metric:N",
                alt.Tooltip(
                    "Score:Q",
                    format=".2f"
                )
            ]
        )
        .properties(height=290)
    )

    st.altair_chart(
        chart,
        width="stretch"
    )

    st.subheader("Night Tracking Diagnostics")

    st.dataframe(
        night[
            [
                "Model",
                "Tracked_Boxes",
                "TP", "FP", "FN",
                "Precision", "Recall",
                "Box_F1", "MOTA", "IDF1",
                "ID_Switches", "Fragments"
            ]
        ].round(2),
        hide_index=True
    )

    st.warning(
        "V2 improved IDF1 and MOTA on this night "
        "validation sequence, but its recall fell "
        "from 29.60% to 28.45%. Night IDF1 remains "
        "low (33.01%). The video was part of the "
        "validation split used during development; "
        "this is not a fresh blind test."
    )

# =====================================
# TAB 3: Data Explorer
# =====================================

with tab3:

    st.subheader("Frame-Level Dataset")

    st.dataframe(
        filtered,
        hide_index=True
    )

    st.download_button(
        label="Download Filtered CSV",
        data=filtered.to_csv(index=False),
        file_name="traffic_dashboard_export.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Dataset Field Definitions")

    st.markdown("""
- **GT_All_Objects:** Total annotated object observations.
- **GT_Cars:** Annotated car observations.
- **YOLO_Detections:** Car detection boxes produced by YOLO.
- **YOLO_TP:** YOLO boxes matched to GT cars at IoU ≥ 0.5.
- **ByteTrack_Outputs:** Reported tracking boxes, including possible false positives.
    """)

# =====================================
# 8. Footer
# =====================================

st.divider()

st.caption(
    "AI Traffic Analytics | EMT Dataset | "
    "YOLO11s + ByteTrack | Research Prototype"
)
