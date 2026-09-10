import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from textwrap import dedent

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Iris Intelligence",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    dedent("""
    <style>
        /* Fix Streamlit Material Icons */
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
    }
    .material-symbols-rounded {
        font-family: "Material Symbols Rounded" !important;
    }

    /* ======================================================
       GLOBAL
    ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 8%,
                rgba(184, 112, 255, 0.13),
                transparent 24%
            ),
            radial-gradient(
                circle at 5% 55%,
                rgba(63, 190, 145, 0.08),
                transparent 24%
            ),
            #080a10;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0c0f17 0%,
                #090b11 100%
            );
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] * {
        font-family: "Segoe UI", sans-serif;
    }


    /* ======================================================
       HERO
    ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 3.6rem 3.5rem;

        border-radius: 30px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.08),
                rgba(255,255,255,0.025)
            );

        border: 1px solid rgba(255,255,255,0.09);

        box-shadow:
            0 25px 80px rgba(0,0,0,0.35);

        margin-bottom: 2rem;
    }

    .hero::before {
        content: "";
        position: absolute;

        width: 360px;
        height: 360px;

        right: -110px;
        top: -160px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(195,110,255,0.35),
                transparent 68%
            );

        filter: blur(25px);
    }

    .hero::after {
        content: "";
        position: absolute;

        width: 250px;
        height: 250px;

        left: -120px;
        bottom: -150px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(64,212,164,0.17),
                transparent 70%
            );

        filter: blur(25px);
    }

    .hero-content {
        position: relative;
        z-index: 2;
    }

    .eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        font-weight: 700;

        opacity: 0.5;

        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-family: Georgia, serif;

        font-size: 4.6rem;
        line-height: 0.98;

        margin: 0;

        letter-spacing: -0.04em;
    }

    .hero-title span {
        background:
            linear-gradient(
                90deg,
                #f3b6ff,
                #a87cff,
                #72e7bf
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 760px;

        margin-top: 1.15rem;

        font-size: 1.08rem;

        line-height: 1.75;

        opacity: 0.66;
    }

    .hero-tags {
        margin-top: 1.5rem;
    }

    .hero-tag {
        display: inline-block;

        padding: 0.48rem 0.85rem;

        margin-right: 0.45rem;

        border-radius: 999px;

        background:
            rgba(255,255,255,0.055);

        border:
            1px solid rgba(255,255,255,0.09);

        font-size: 0.78rem;

        opacity: 0.82;
    }


    /* ======================================================
       SECTION HEADERS
    ====================================================== */

    .section-eyebrow {
        margin-top: 1.8rem;

        font-size: 0.7rem;

        text-transform: uppercase;

        letter-spacing: 0.18em;

        font-weight: 700;

        opacity: 0.42;
    }

    .section-title {
        font-family: Georgia, serif;

        font-size: 2.15rem;

        margin-top: 0.15rem;

        margin-bottom: 1rem;
    }


    /* ======================================================
       KPI CARDS
    ====================================================== */

    .kpi {
        min-height: 140px;

        padding: 1.45rem;

        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.055),
                rgba(255,255,255,0.018)
            );

        border: 1px solid rgba(255,255,255,0.075);

        box-shadow:
            0 14px 35px rgba(0,0,0,0.15);
    }

    .kpi-icon {
        font-size: 1.25rem;
        margin-bottom: 0.55rem;
    }

    .kpi-label {
        font-size: 0.72rem;

        text-transform: uppercase;

        letter-spacing: 0.1em;

        opacity: 0.45;
    }

    .kpi-value {
        font-size: 2.05rem;

        font-weight: 700;

        margin-top: 0.3rem;
    }

    .kpi-description {
        font-size: 0.76rem;

        opacity: 0.42;

        margin-top: 0.25rem;
    }


    /* ======================================================
       PREDICTION PANEL
    ====================================================== */

    .prediction-panel {
        padding: 2rem;

        border-radius: 25px;

        background:
            linear-gradient(
                145deg,
                rgba(155,92,232,0.13),
                rgba(43,185,137,0.055)
            );

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 22px 60px rgba(0,0,0,0.22);
    }

    .prediction-result {
        margin-top: 1.4rem;

        padding: 2.1rem;

        border-radius: 22px;

        text-align: center;

        background:
            rgba(255,255,255,0.04);

        border:
            1px solid rgba(255,255,255,0.09);
    }

    .prediction-caption {
        font-size: 0.68rem;

        letter-spacing: 0.18em;

        text-transform: uppercase;

        opacity: 0.45;
    }

    .prediction-name {
        font-family: Georgia, serif;

        font-size: 2.65rem;

        margin-top: 0.25rem;
    }

    .prediction-confidence {
        font-size: 0.95rem;

        opacity: 0.62;

        margin-top: 0.4rem;
    }


    /* ======================================================
       SPECIES CARDS
    ====================================================== */

    .species-card {
        min-height: 185px;

        padding: 1.5rem;

        border-radius: 20px;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid rgba(255,255,255,0.075);
    }

    .species-icon {
        font-size: 2rem;
    }

    .species-title {
        font-size: 1.1rem;

        font-weight: 700;

        margin-top: 0.65rem;
    }

    .species-copy {
        font-size: 0.82rem;

        line-height: 1.55;

        opacity: 0.47;

        margin-top: 0.35rem;
    }


    /* ======================================================
       INFO CARDS
    ====================================================== */

    .info-card {
        padding: 1.35rem 1.45rem;

        border-radius: 18px;

        background:
            rgba(255,255,255,0.03);

        border:
            1px solid rgba(255,255,255,0.07);
    }

    .info-label {
        font-size: 0.68rem;

        text-transform: uppercase;

        letter-spacing: 0.12em;

        opacity: 0.4;
    }

    .info-value {
        margin-top: 0.4rem;

        font-size: 1rem;

        font-weight: 600;
    }


    /* ======================================================
       BUTTON
    ====================================================== */

    .stButton > button {
        border-radius: 13px !important;

        border: 1px solid rgba(255,255,255,0.13) !important;

        background:
            linear-gradient(
                90deg,
                rgba(174,99,244,0.35),
                rgba(59,193,151,0.25)
            ) !important;

        font-weight: 700 !important;

        min-height: 3rem !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 10px 28px rgba(160,100,255,0.17);
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        margin-top: 4rem;

        padding-top: 1.5rem;

        border-top:
            1px solid rgba(255,255,255,0.07);

        text-align: center;

        opacity: 0.38;

        font-size: 0.76rem;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# DATA
# ============================================================

df = pd.read_csv("iris.csv")

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]


# ============================================================
# MACHINE LEARNING
# ============================================================

X = df[features]
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌸 Iris Intelligence")

    st.caption(
        "Machine Learning • Botanical Analytics"
    )

    st.divider()

    st.markdown("### Navigation")

    section = st.radio(
        "Explore",
        [
            "Overview",
            "Predict",
            "Analytics",
            "Model Intelligence"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model")

    st.write("Random Forest Classifier")

    st.markdown("### Dataset")

    st.write("150 observations")

    st.markdown("### Classes")

    st.write("Setosa · Versicolor · Virginica")

    st.divider()

    st.caption(
        "Python · Pandas · Scikit-learn · Plotly · Streamlit"
    )


# ============================================================
# HERO
# ============================================================

st.markdown("### 🌿 MACHINE LEARNING · BOTANICAL INTELLIGENCE")

st.title("Discover Iris Intelligence.")

st.markdown(
    """
    Explore the geometry of the Iris flower and use machine learning
    to identify its species from four simple measurements.
    """
)

st.caption(
    "🌿 Random Forest  ·  📊 Interactive Analytics  ·  🔮 Real-time Prediction"
)


# ============================================================
# OVERVIEW
# ============================================================

if section == "Overview":

    st.markdown(
        '<div class="section-eyebrow">PROJECT SNAPSHOT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">The dataset behind the intelligence</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)


    with c1:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">🌱</div>
                <div class="kpi-label">Observations</div>
                <div class="kpi-value">{len(df)}</div>
                <div class="kpi-description">Flower measurements</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">🌺</div>
                <div class="kpi-label">Species</div>
                <div class="kpi-value">{df["species"].nunique()}</div>
                <div class="kpi-description">Classes in dataset</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">Accuracy</div>
                <div class="kpi-value">{accuracy:.1%}</div>
                <div class="kpi-description">Test-set performance</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">📐</div>
                <div class="kpi-label">Features</div>
                <div class="kpi-value">{len(features)}</div>
                <div class="kpi-description">Flower dimensions</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()

    st.markdown(
        '<div class="section-eyebrow">SPECIES GUIDE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Three flowers. One classification challenge.</div>',
        unsafe_allow_html=True
    )


    s1, s2, s3 = st.columns(3)


    with s1:
        st.markdown(
            """
            <div class="species-card">
                <div class="species-icon">🌱</div>
                <div class="species-title">Iris Setosa</div>
                <div class="species-copy">
                    The smallest-petal profile in the dataset,
                    with measurements that make it highly distinct.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with s2:
        st.markdown(
            """
            <div class="species-card">
                <div class="species-icon">🌷</div>
                <div class="species-title">Iris Versicolor</div>
                <div class="species-copy">
                    The intermediate class, occupying the
                    measurement space between the other species.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with s3:
        st.markdown(
            """
            <div class="species-card">
                <div class="species-icon">🌺</div>
                <div class="species-title">Iris Virginica</div>
                <div class="species-copy">
                    Distinguished by comparatively larger
                    petals and a distinctive measurement profile.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()

    st.markdown(
        '<div class="section-eyebrow">FLOWER GEOMETRY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">See the species separate in feature space</div>',
        unsafe_allow_html=True
    )


    g1, g2 = st.columns(2)


    with g1:

        fig = px.scatter(
            df,
            x="sepal_length",
            y="sepal_width",
            color="species",
            size="petal_width",
            hover_data=[
                "petal_length"
            ],
            title="Sepal Geometry"
        )

        fig.update_layout(
            height=430,
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with g2:

        fig = px.scatter(
            df,
            x="petal_length",
            y="petal_width",
            color="species",
            size="sepal_length",
            hover_data=[
                "sepal_width"
            ],
            title="Petal Geometry"
        )

        fig.update_layout(
            height=430,
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PREDICTION
# ============================================================
elif section == "Predict":

    st.markdown(
        '<div class="section-eyebrow">PREDICTION STUDIO</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Let the model identify your flower</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter four measurements. The trained Random Forest model will estimate "
        "the most likely Iris species and show how confident it is."
    )

    st.write("")

    # --------------------------------------------------------
    # INPUT PANEL
    # --------------------------------------------------------

    st.markdown(
        dedent("""
        <div class="prediction-panel">
        """),
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.1,
            step=0.1
        )

    with p2:
        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=5.0,
            value=3.5,
            step=0.1
        )

    with p3:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=1.4,
            step=0.1
        )

    with p4:
        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=3.0,
            value=0.2,
            step=0.1
        )

    st.write("")

    predict_button = st.button(
        "🌸  IDENTIFY SPECIES",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = probabilities.max()

        # Result
        st.markdown(
            dedent(f"""
            <div class="prediction-result">

                <div class="prediction-caption">
                    MODEL PREDICTION
                </div>

                <div class="prediction-name">
                    🌸 Iris-{prediction.title()}
                </div>

                <div class="prediction-confidence">
                    Model confidence · <strong>{confidence:.1%}</strong>
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )

        st.write("")

        # ----------------------------------------------------
        # PROBABILITIES + GAUGE
        # ----------------------------------------------------

        left, right = st.columns(2)

        with left:

            probability_df = pd.DataFrame({
                "Species": [
                    "Setosa",
                    "Versicolor",
                    "Virginica"
                ],
                "Probability": probabilities
            })

            probability_fig = px.bar(
                probability_df,
                x="Probability",
                y="Species",
                orientation="h",
                text=probability_df["Probability"].map(
                    lambda x: f"{x:.1%}"
                ),
                title="Species Probability"
            )

            probability_fig.update_traces(
                textposition="outside"
            )

            probability_fig.update_layout(
                template="plotly_dark",
                height=380,
                xaxis_tickformat=".0%",
                xaxis_title="Probability",
                yaxis_title=""
            )

            st.plotly_chart(
                probability_fig,
                use_container_width=True
            )

        with right:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={
                        "text": "Prediction Confidence"
                    },
                    number={
                        "suffix": "%"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        }
                    }
                )
            )

            gauge.update_layout(
                template="plotly_dark",
                height=380
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        # ----------------------------------------------------
        # MEASUREMENT PROFILE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-eyebrow">MEASUREMENT PROFILE</div>',
            unsafe_allow_html=True
        )

        profile_df = pd.DataFrame({
            "Measurement": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width"
            ],
            "Value": [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        })

        profile_fig = px.bar(
            profile_df,
            x="Measurement",
            y="Value",
            title="Your Flower's Measurements"
        )

        profile_fig.update_layout(
            template="plotly_dark",
            height=380
        )

        st.plotly_chart(
            profile_fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # MODEL INTERPRETATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-eyebrow">QUICK INTERPRETATION</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            dedent(f"""
            <div class="info-card">

                <div class="info-label">
                    RESULT
                </div>

                <div class="info-value">
                    The model predicts <strong>Iris-{prediction.title()}</strong>
                    with a confidence of <strong>{confidence:.1%}</strong>.
                </div>

            </div>
            """),
            unsafe_allow_html=True
        )

# ============================================================
# ANALYTICS
# ============================================================

elif section == "Analytics":

    st.markdown(
        '<div class="section-eyebrow">EXPLORATORY ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Explore the anatomy of Iris</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Interact with the measurements to discover how the three species "
        "separate across feature space."
    )

    # --------------------------------------------------------
    # SPECIES COUNTS
    # --------------------------------------------------------

    species_counts = (
        df["species"]
        .value_counts()
        .reset_index()
    )

    species_counts.columns = ["Species", "Count"]

    c1, c2 = st.columns(2)

    with c1:

        fig = px.bar(
            species_counts,
            x="Species",
            y="Count",
            color="Species",
            text="Count",
            title="Samples by Species"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.pie(
            species_counts,
            names="Species",
            values="Count",
            hole=0.55,
            title="Species Composition"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE RELATIONSHIP
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">FEATURE RELATIONSHIPS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Watch the species separate</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        x_feature = st.selectbox(
            "X-axis feature",
            features,
            index=0,
            format_func=lambda x: x.replace("_", " ").title()
        )

    with col2:

        y_feature = st.selectbox(
            "Y-axis feature",
            features,
            index=2,
            format_func=lambda x: x.replace("_", " ").title()
        )

    scatter_fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="species",
        size="petal_width",
        hover_data=features,
        title=f"{x_feature.replace('_', ' ').title()} vs "
              f"{y_feature.replace('_', ' ').title()}"
    )

    scatter_fig.update_layout(
        template="plotly_dark",
        height=520
    )

    st.plotly_chart(
        scatter_fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">DISTRIBUTION LAB</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Compare measurement ranges</div>',
        unsafe_allow_html=True
    )

    selected_feature = st.selectbox(
        "Choose a measurement",
        features,
        format_func=lambda x: x.replace("_", " ").title()
    )

    d1, d2 = st.columns(2)

    with d1:

        hist_fig = px.histogram(
            df,
            x=selected_feature,
            color="species",
            marginal="box",
            barmode="overlay",
            opacity=0.75,
            title=f"{selected_feature.replace('_', ' ').title()} Distribution"
        )

        hist_fig.update_layout(
            template="plotly_dark",
            height=430
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

    with d2:

        box_fig = px.box(
            df,
            x="species",
            y=selected_feature,
            color="species",
            points="all",
            title=f"{selected_feature.replace('_', ' ').title()} by Species"
        )

        box_fig.update_layout(
            template="plotly_dark",
            height=430
        )

        st.plotly_chart(
            box_fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">CORRELATION MAP</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">How strongly do the measurements relate?</div>',
        unsafe_allow_html=True
    )

    correlation = df[features].corr()

    corr_fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    corr_fig.update_layout(
        template="plotly_dark",
        height=520
    )

    st.plotly_chart(
        corr_fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">STATISTICAL SNAPSHOT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Species-level measurement summary</div>',
        unsafe_allow_html=True
    )

    summary = (
        df.groupby("species")[features]
        .mean()
        .round(2)
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif section == "Model Intelligence":

    st.markdown(
        '<div class="section-eyebrow">MODEL INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Inside the classifier</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Understand how the Random Forest model performs and which "
        "measurements contribute most to its decisions."
    )

    # --------------------------------------------------------
    # MODEL KPIs
    # --------------------------------------------------------

    accuracy_value = accuracy * 100

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Accuracy",
            f"{accuracy_value:.1f}%"
        )

    with k2:
        st.metric(
            "Precision",
            f"{macro_precision:.1%}"
        )

    with k3:
        st.metric(
            "Recall",
            f"{macro_recall:.1%}"
        )

    with k4:
        st.metric(
            "F1 Score",
            f"{macro_f1:.1%}"
        )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    a, b = st.columns(2)

    with a:

        st.markdown("### Confusion Matrix")

        cm_fig = px.imshow(
            cm,
            text_auto=True,
            x=[
                label.title()
                for label in model.classes_
            ],
            y=[
                label.title()
                for label in model.classes_
            ],
            labels={
                "x": "Predicted Species",
                "y": "Actual Species",
                "color": "Count"
            },
            title="Where the predictions land"
        )

        cm_fig.update_layout(
            template="plotly_dark",
            height=480
        )

        st.plotly_chart(
            cm_fig,
            use_container_width=True
        )

    with b:

        st.markdown("### Classification Report")

        clean_report = report_df.loc[
            ["setosa", "versicolor", "virginica"]
        ][
            ["precision", "recall", "f1-score", "support"]
        ].round(2)

        clean_report.index = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]

        st.dataframe(
            clean_report,
            use_container_width=True,
            height=430
        )

        st.info(
            "Precision measures how reliable each predicted class is, "
            "while recall measures how many actual samples of that class "
            "were successfully identified."
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">MODEL EXPLAINABILITY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">What drives the prediction?</div>',
        unsafe_allow_html=True
    )

    importance_df = pd.DataFrame({
        "Feature": [
            feature.replace("_", " ").title()
            for feature in features
        ],
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=importance_df["Importance"].map(
            lambda x: f"{x:.1%}"
        ),
        title="Random Forest Feature Importance"
    )

    importance_fig.update_traces(
        textposition="outside"
    )

    importance_fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_tickformat=".0%"
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL ARCHITECTURE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">MODEL ARCHITECTURE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">How Iris Intelligence works</div>',
        unsafe_allow_html=True
    )

    pipeline1, pipeline2, pipeline3, pipeline4 = st.columns(4)

    with pipeline1:
        st.info(
            "01\n\n"
            "INPUT\n\n"
            "Four flower measurements"
        )

    with pipeline2:
        st.info(
            "02\n\n"
            "TRAIN\n\n"
            "Random Forest learns patterns"
        )

    with pipeline3:
        st.info(
            "03\n\n"
            "PREDICT\n\n"
            "Model estimates species probabilities"
        )

    with pipeline4:
        st.info(
            "04\n\n"
            "EXPLAIN\n\n"
            "Metrics reveal model behavior"
        )
# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌸 Iris Intelligence
        <br><br>
        Machine Learning Portfolio Project ·
        Random Forest Classification ·
        Interactive Data Visualization
    </div>
    """,
    unsafe_allow_html=True
)