import streamlit as st
from PIL import Image
from modules.image.pipeline import analyze_image
import tempfile
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VeriMedia AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN APP ---------- */

    .stApp {
        background-color: #0b1020;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HERO ---------- */

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827,
            #172554
        );
        border: 1px solid #263452;
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #aeb8cc;
        margin-top: 6px;
    }


    /* ---------- CARDS ---------- */

    .card {
        padding: 22px;
        border-radius: 16px;
        background: #111827;
        border: 1px solid #263452;
        margin-bottom: 18px;
    }


    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        padding: 22px;
        border-radius: 16px;
        background: #111827;
        border: 1px solid #263452;
        text-align: center;
        min-height: 125px;
    }

    .metric-title {
        color: #9ca3af;
        font-size: 14px;
        font-weight: 500;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 27px;
        font-weight: 800;
        margin-top: 10px;
    }


    /* ---------- FORENSIC STATUS ---------- */

    .status-card {
        padding: 22px;
        border-radius: 16px;
        background: #111827;
        border: 1px solid #263452;
        line-height: 1.7;
    }


    /* ---------- INFORMATION BOX ---------- */

    .info-card {
        padding: 18px;
        border-radius: 14px;
        background: #172033;
        border: 1px solid #30405f;
        line-height: 1.7;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #1f2937;
        font-size: 13px;
    }


    /* ---------- ANALYSIS HEADER ---------- */

    .analysis-header {
        padding: 18px 22px;
        border-radius: 14px;
        background: #111827;
        border: 1px solid #263452;
        margin-bottom: 20px;
    }

    .analysis-id {
        color: #94a3b8;
        font-size: 13px;
    }


    /* ---------- DISCLAIMER ---------- */

    .disclaimer {
        padding: 15px;
        border-radius: 12px;
        background: #1c1917;
        border: 1px solid #44403c;
        color: #d6d3d1;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🔎 VeriMedia AI")

    st.caption("Digital Media Forensics")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Analyze Media",
            "Analysis History",
            "About"
        ]
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.success("● SYSTEM ONLINE")

    st.caption("Forensic Engine: Ready")

    st.divider()

    st.caption("Version 1.0")
    st.caption("B.Tech Major Project")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🔎 VeriMedia AI
        </div>

        <div class="hero-subtitle">
            Multimodal AI System for Digital Media Forensics
            and Authenticity Analysis
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ANALYZE MEDIA PAGE
# ============================================================

if page == "Analyze Media":

    st.markdown("## 🎯 Analyze Media")

    st.write(
        "Upload digital media to analyze indicators associated "
        "with AI-generated or manipulated content."
    )

    st.markdown("### 📁 Upload Image")

    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help="Supported formats: JPG, JPEG, PNG and WEBP."
    )


    # ========================================================
    # FILE UPLOADED
    # ========================================================

    if uploaded_file:

        try:

            image = Image.open(uploaded_file).convert("RGB")

            # ------------------------------------------------
            # ANALYSIS ID
            # ------------------------------------------------

            import hashlib

            file_bytes = uploaded_file.getvalue()

            file_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            analysis_id = "VM-" + file_hash[:8].upper()


            # ------------------------------------------------
            # ANALYSIS HEADER
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="analysis-header">

                    <b>Analysis Session</b>

                    <br>

                    <span class="analysis-id">
                        {analysis_id}
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # PREVIEW + FILE INFORMATION
            # =================================================

            preview_col, info_col = st.columns(
                [1.25, 1]
            )


            # -------------------------------------------------
            # IMAGE PREVIEW
            # -------------------------------------------------

            with preview_col:

                st.markdown("### 📷 Media Preview")

                st.image(
                    image,
                    use_container_width=True
                )


            # -------------------------------------------------
            # MEDIA INFORMATION
            # -------------------------------------------------

            with info_col:

                st.markdown("### 📄 Media Information")

                st.markdown(
                    f"""
                    <div class="card">

                    <b>Filename</b><br>
                    {uploaded_file.name}

                    <br><br>

                    <b>Format</b><br>
                    {uploaded_file.type}

                    <br><br>

                    <b>File Size</b><br>
                    {uploaded_file.size / 1024:.2f} KB

                    <br><br>

                    <b>Resolution</b><br>
                    {image.width} × {image.height}

                    <br><br>

                    <b>SHA-256</b><br>
                    <code>{file_hash[:20]}...</code>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.divider()


            # =================================================
            # ANALYSIS BUTTON
            # =================================================

            st.markdown("### 🧠 Forensic Analysis")

            analyze_button = st.button(
                "🔍 START FORENSIC ANALYSIS",
                use_container_width=True,
                type="primary"
            )


            if analyze_button:

                # -------------------------------------------------
                # PROGRESS
                # -------------------------------------------------

                progress = st.progress(0)

                status = st.empty()


                status.info(
                    "🔄 Initializing forensic engine..."
                )

                progress.progress(15)


                status.info(
                    "🔄 Validating uploaded image..."
                )

                progress.progress(25)


                status.info(
                    "🔄 Preprocessing image..."
                )

                progress.progress(40)


                status.info(
                    "🔄 Running AI-generation detector..."
                )


                # =================================================
                # REAL MODEL
                # =================================================

                # REAL MODEL
                # REAL MODEL — Member 1 Image Forensics Pipeline
                suffix = Path(uploaded_file.name).suffix or ".jpg"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_image_path = temp_file.name

                try:
                    result = analyze_image(
                        temp_image_path,
                        generate_heatmap=True
                    )
                finally:
                    Path(temp_image_path).unlink(
                        missing_ok=True
                    )

                progress.progress(85)


                status.info(
                    "🔄 Preparing forensic assessment..."
                )


                # -------------------------------------------------
                # EXTRACT RESULT
                # -------------------------------------------------

                prediction = result["ai_detection"]["predicted_label"]

                ai_probability = result["ai_detection"]["ai_generated_probability"]
                real_probability = result["ai_detection"]["real_probability"]

                ai_percentage = ai_probability * 100
                real_percentage = real_probability * 100

                progress.progress(100)
                                # -------------------------------------------------
                # FORENSIC INDICATORS
                # -------------------------------------------------

                metadata = result["metadata"]
                compression = result["compression"]
                noise = result["noise"]
                frequency = result["frequency"]

                st.subheader("🔬 Image Forensic Indicators")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Image Format",
                        metadata["format"]
                    )
                    st.metric(
                        "Resolution",
                        f'{metadata["width"]} × {metadata["height"]}'
                    )

                with col2:
                    st.metric(
                        "File Size",
                        f'{metadata["file_size_bytes"]:,} bytes'
                    )
                    st.metric(
                        "Noise Std.",
                        f'{noise["noise_std"]:.2f}'
                    )

                with col3:
                    st.metric(
                        "High-Frequency Ratio",
                        f'{frequency["high_frequency_ratio"]:.2%}'
                    )
                    st.metric(
                        "JPEG Quantization",
                        "Detected"
                        if compression["jpeg_quantization_tables"]
                        else "Not detected"
                    )

                # -------------------------------------------------
                # SUPPORTING EVIDENCE
                # -------------------------------------------------

                st.subheader("📋 Supporting Forensic Evidence")

                evidence_items = []

                evidence_items.extend(
                    compression.get("evidence", [])
                )

                evidence_items.extend(
                    noise.get("evidence", [])
                )

                evidence_items.extend(
                    frequency.get("evidence", [])
                )

                if evidence_items:
                    for item in evidence_items:
                        st.write(f"• {item}")
                else:
                    st.info("No supporting forensic indicators available.")


                status.success(
                    "✅ Forensic analysis completed."
                )


                st.divider()


                # =================================================
                # AUTHENTICITY ASSESSMENT
                # =================================================

                st.markdown(
                    "## 🧠 Authenticity Assessment"
                )


                                # -------------------------------------------------
                # FORENSIC ASSESSMENT
                # -------------------------------------------------
                # Initial heuristic thresholds for UI interpretation.
                # These values are not calibrated forensic probabilities
                # and must be validated against the project evaluation dataset.
                if ai_probability >= 0.80:
                    assessment = "High AI-generation likelihood"
                    icon = "⚠️"

                elif ai_probability >= 0.50:

                    assessment = "Elevated AI-generation likelihood"
                    icon = "⚠️"

                else:

                    assessment = "Lower AI-generation likelihood"
                    icon = "✅"
                                # -------------------------------------------------
                # METRIC CARDS
                # -------------------------------------------------

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                AI-Generation Score
                            </div>
                            <div class="metric-value">
                                {ai_percentage:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Real-Image Score
                            </div>
                            <div class="metric-value">
                                {real_percentage:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Assessment
                            </div>
                            <div class="metric-value">
                                {icon} {assessment}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # SCORE VISUALIZATION
                # =================================================

                st.markdown(
                    "### 📊 AI-Generation Probability"
                )

                st.progress(
                    min(
                        max(ai_probability, 0.0),
                        1.0
                    )
                )

                st.caption(
                    f"AI-generation model score: "
                    f"{ai_percentage:.2f}%"
                )


                # =================================================
                # FORENSIC INTERPRETATION
                # =================================================

                st.markdown(
                    "### 🔬 Forensic Interpretation"
                )


                if ai_probability >= 0.80:

                    interpretation = (
                        "The image received a high AI-generation "
                        "score from the baseline detector. This "
                        "indicates that the model identified "
                        "patterns associated with synthetic imagery. "
                        "Additional forensic evidence is required "
                        "before making a final authenticity decision."
                    )

                elif ai_probability >= 0.50:

                    interpretation = (
                        "The image received an elevated AI-generation "
                        "score from the baseline detector. This result "
                        "should be treated as an indicator rather than "
                        "proof of AI generation and should be evaluated "
                        "alongside other forensic evidence."
                    )

                else:

                    interpretation = (
                        "The image received a relatively low "
                        "AI-generation score from the baseline detector. "
                        "The result is more consistent with the "
                        "real-image class according to this model, "
                        "but it does not by itself prove authenticity."
                    )


                st.markdown(
                    f"""
                    <div class="status-card">

                    <b>Assessment:</b>
                    {assessment}

                    <br><br>

                    <b>Model interpretation:</b>

                    <br>

                    {interpretation}

                    <br><br>

                    <b>AI-generation score:</b>
                    {ai_percentage:.2f}%

                    <br>

                    <b>Real-image score:</b>
                    {real_percentage:.2f}%

                    </div>
                    """,
                    unsafe_allow_html=True
                )
                                # =================================================
                # FORENSIC RESIDUAL HEATMAP
                # =================================================

                explainability = result.get("explainability")

                if explainability:

                    st.markdown(
                        "### 🗺️ Forensic Residual Heatmap"
                    )

                    heatmap_path = explainability.get(
                        "heatmap_path"
                    )

                    if heatmap_path and Path(heatmap_path).exists():

                        st.image(
                            heatmap_path,
                            caption=(
                                "Residual heatmap showing regions "
                                "with stronger local residual activity."
                            ),
                            use_container_width=True
                        )

                        st.info(
                            "⚠️ This visualization highlights "
                            "local noise/residual activity. It is "
                            "supporting forensic evidence and should "
                            "not be interpreted as proof of AI "
                            "generation or manipulation."
                        )

                    else:

                        st.warning(
                            "Forensic heatmap could not be generated."
                        )


                # =================================================
                # EVIDENCE SECTION
                # =================================================

                st.markdown(
                    "### 🔬 Forensic Evidence"
                )


                evidence_col1, evidence_col2 = st.columns(2)


                with evidence_col1:

                    st.markdown(
                        f"""
                        <div class="card">

                        <b>Visual Classification</b>

                        <br><br>

                        AI-generated score:
                        <b>{ai_percentage:.1f}%</b>

                        <br>

                        Real-image score:
                        <b>{real_percentage:.1f}%</b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with evidence_col2:

                    st.markdown(
                        f"""
                        <div class="card">

                        <b>File Integrity</b>

                        <br><br>

                        SHA-256 hash generated

                        <br>

                        <code>
                        {file_hash[:32]}...
                        </code>

                        <br><br>

                        Used to identify the analyzed file.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # MODEL INFORMATION
                # =================================================

                with st.expander(
                    "🧠 View Detection Model Information"
                ):

                    st.write(
                        "**Model:** "
                        "SteganographIA AI Image Detector"
                    )

                    st.write(
                        "**Architecture:** "
                        "Vision Transformer (ViT)"
                    )

                    st.write(
                        "**Task:** "
                        "Real vs AI-generated image classification"
                    )

                    st.write(
                        "**Framework:** "
                        "PyTorch + Hugging Face Transformers"
                    )

                    st.write(
                        "**Output:** "
                        "Real and AI-generated class scores"
                    )


                # =================================================
                # LIMITATION
                # =================================================

                st.markdown(
                    f"""
                    <div class="disclaimer">

                    ⚠️ <b>Important:</b>
                    VeriMedia AI provides a model-based
                    forensic estimate. A high or low score
                    does not constitute definitive proof
                    that an image is authentic or manipulated.

                    Results should be considered alongside
                    additional forensic evidence and human
                    verification.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        except Exception as e:

            st.error(
                "❌ Analysis failed."
            )

            st.write(
                "Technical details:"
            )

            st.exception(e)


# ============================================================
# ANALYSIS HISTORY
# ============================================================

elif page == "Analysis History":

    st.markdown("## 📊 Analysis History")

    st.info(
        "Analysis history will be connected after "
        "the database module is implemented."
    )

    st.markdown(
        """
        ### Planned History Features

        - Analysis ID
        - Filename
        - Media type
        - Detection result
        - Risk score
        - Date and time
        - Report download
        """
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown("## ℹ️ About VeriMedia AI")

    st.write(
        """
        VeriMedia AI is a multimodal digital-media
        forensics research system designed to estimate
        whether digital media may be AI-generated or
        manipulated.
        """
    )


    st.markdown("### 🎯 Project Objective")

    st.write(
        """
        The system combines specialized forensic analysis
        techniques across image, video and audio modalities
        to provide evidence-based authenticity assessment.
        """
    )


    st.markdown("### 🧩 Detection Modules")

    st.markdown(
        """
        **🖼️ Image Forensics**

        Detect indicators associated with AI-generated imagery.

        **🎥 Video Forensics**

        Analyze frames, faces and temporal inconsistencies.

        **🎙️ Audio Forensics**

        Analyze possible synthetic or manipulated speech.

        **👄 Lip-Sync Analysis**

        Compare facial movement with audio timing.

        **🧠 Multimodal Fusion**

        Combine evidence from multiple modalities.

        **🤖 DeepSeek Reasoning**

        Convert structured forensic evidence into
        human-readable explanations.

        **📄 Forensic Reporting**

        Generate structured forensic reports.
        """
    )


    st.markdown("### 🔬 Research Question")

    st.info(
        """
        Does combining evidence from multiple media
        modalities provide a more reliable authenticity
        assessment than analyzing a single modality?
        """
    )


    st.warning(
        """
        VeriMedia AI is a decision-support system.
        It does not claim to provide definitive proof
        of authenticity or manipulation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        VeriMedia AI • Digital Media Forensics Research System

        <br>

        B.Tech / B.E. Major Project

    </div>
    """,
    unsafe_allow_html=True
)