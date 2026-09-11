# VeriMedia AI

## A Multimodal AI System for Digital Media Forensics and Authenticity Analysis

## 1. Problem Statement

The rapid growth of Generative AI and advanced editing technologies has made it difficult to distinguish authentic digital media from AI-generated or manipulated content.

Modern technologies can create or modify:

* AI-generated images
* Deepfake videos
* Face-swapped videos
* Synthetic or cloned voices
* Manipulated audio
* Lip-sync manipulated videos
* AI-edited photographs

A single detection method may not always provide reliable results. Therefore, VeriMedia AI aims to analyze multiple forensic signals and combine them to provide an evidence-based assessment of media authenticity.

---

## 2. Project Objective

The main objective of VeriMedia AI is to develop an AI-assisted digital media forensic system that can analyze images, videos, and audio to estimate the likelihood of manipulation or synthetic generation.

The system aims to:

* Detect AI-generated images
* Detect manipulated images
* Analyze videos for deepfake indicators
* Detect synthetic or cloned speech
* Analyze audio-video synchronization
* Identify suspicious image regions
* Identify suspicious video frames and timestamps
* Combine evidence from multiple forensic modules
* Provide an evidence-based authenticity assessment
* Generate human-readable forensic explanations
* Generate structured forensic reports
* Evaluate whether multimodal analysis performs better than single-modality analysis

---

## 3. Research Question

The primary research question of VeriMedia AI is:

> Does combining multiple forensic modalities provide a more reliable digital media authenticity assessment than analyzing a single modality?

The project will compare:

* Visual-only analysis
* Audio-only analysis
* Visual + Audio analysis
* Visual + Audio + Lip-Sync analysis
* Different evidence-fusion strategies

The system will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* False Positive Rate
* False Negative Rate

The research will determine whether combining multiple independent forensic signals can improve the reliability and robustness of digital media authenticity assessment.

---

## 4. System Architecture

The proposed VeriMedia AI pipeline follows this workflow:

**User Upload → Media Identification → Preprocessing → Specialized Forensic Analysis → Evidence Extraction → Multimodal Fusion → Risk Assessment → DeepSeek Explanation → Final Forensic Report**

### High-Level Architecture

```text
User Upload
     |
     v
Media Identification
     |
     v
Preprocessing
     |
     +-------------------+-------------------+
     |                   |                   |
     v                   v                   v
Image Forensics     Video Forensics     Audio Forensics
     |                   |                   |
     |                   v                   |
     |              Lip-Sync Analysis       |
     |                   |                   |
     +-------------------+-------------------+
                         |
                         v
                 Evidence Fusion
                         |
                         v
                  Risk Assessment
                         |
                         v
                DeepSeek Explanation
                         |
                         v
                 Forensic Report
```

### Core Principle

DeepSeek is **not the primary detection model**.

The specialized forensic modules perform the actual analysis and generate evidence. DeepSeek receives the structured forensic evidence and converts it into a clear, human-readable explanation and report.

This modular architecture allows each component to be developed, tested, evaluated, and improved independently.

## 5. Major Modules

### Image Forensics

The image module analyzes:

* AI-generation indicators
* Metadata and EXIF information
* Compression characteristics
* Noise and residual patterns
* Frequency-domain characteristics
* Suspicious image regions
* Explainability information

### Video Forensics

The video module analyzes:

* Video metadata
* Video frames
* Face detection
* Deepfake indicators
* Frame-level predictions
* Temporal inconsistencies
* Suspicious timestamps

### Audio Forensics

The audio module analyzes:

* Synthetic speech indicators
* Voice-cloning indicators
* Spectral characteristics
* Audio manipulation artifacts

### Lip-Sync Analysis

The lip-sync module analyzes:

* Face landmarks
* Mouth movement
* Speech timing
* Audio-video synchronization
* Potential synchronization mismatches

A synchronization mismatch is treated as **forensic evidence**, not automatic proof of manipulation.

### Evidence Fusion

The fusion layer combines evidence from multiple forensic modules.

The project will experiment with different fusion methods, including:

* Equal-weight fusion
* Expert-defined weights
* Logistic regression
* Random forest
* Other experimentally validated methods

### DeepSeek Explanation Layer

DeepSeek receives structured forensic evidence and converts it into a human-readable explanation.

**DeepSeek is not the primary detector.**

The specialized forensic modules generate the evidence, while DeepSeek acts as the **explanation and reporting layer**.

---

## 6. Authenticity Assessment

The system will not treat a single detector score as definitive proof.

| Assessment       | Meaning                                                 |
| ---------------- | ------------------------------------------------------- |
| Low Concern      | Limited forensic indicators detected                    |
| Inconclusive     | Evidence is insufficient or conflicting                 |
| Elevated Concern | Multiple indicators require attention                   |
| High Concern     | Strong evidence of manipulation or synthetic generation |

When forensic signals strongly disagree, the system may return **Inconclusive** instead of forcing a Real/Fake decision.

---

## 7. Technology Stack

### Programming

* Python

### Machine Learning

* PyTorch
* Hugging Face Transformers
* Scikit-learn

### Image Processing

* Pillow
* OpenCV
* NumPy
* SciPy

### Video Processing

* OpenCV
* FFmpeg
* PyTorch

### Audio Processing

* Librosa
* Torchaudio
* NumPy
* SciPy

### Frontend

* Streamlit

### Backend

* FastAPI
* Pydantic

### Database

* PostgreSQL / MySQL
* SQLAlchemy

### AI Explanation

* DeepSeek API

### Reporting

* ReportLab

### Development

* Git
* GitHub
* Docker
* Pytest

## 8. Project Structure

```text
VeriMediaAI/
│
├── frontend/
│   └── streamlit_app.py
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── schemas/
│   └── services/
│
├── modules/
│   ├── image/
│   ├── video/
│   ├── audio/
│   ├── lipsync/
│   └── fusion/
│
├── ai/
│   └── deepseek.py
│
├── database/
│   ├── models.py
│   └── connection.py
│
├── evaluation/
│   ├── image_eval.py
│   ├── video_eval.py
│   ├── audio_eval.py
│   └── fusion_eval.py
│
├── reports/
│   └── generator.py
│
├── tests/
│
├── models/
├── uploads/
├── results/
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── integration_contract.md
│   └── evaluation.md
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
## 9. Team Responsibilities

### Member 1 — Image Forensics

Responsible for:

* Image detector benchmarking
* Image preprocessing
* Metadata analysis
* Compression analysis
* Noise analysis
* Frequency analysis
* Explainability
* Image evaluation

### Member 2 — Video Forensics

Responsible for:

* Video preprocessing
* Frame extraction
* Face detection
* Deepfake detection
* Temporal analysis
* Suspicious timestamps
* Video evaluation

### Member 3 — Audio & Lip-Sync

Responsible for:

* Audio preprocessing
* Spectrogram analysis
* Synthetic speech detection
* Face and lip landmarks
* Mouth tracking
* Audio-video alignment
* Lip-sync scoring

### Member 4 — Backend, Fusion & DeepSeek

Responsible for:

* Backend APIs
* Database
* Module orchestration
* Evidence fusion
* Risk engine
* DeepSeek integration
* Report generation
* System integration

---

## 10. Common Integration Contract

Each forensic module should return structured data similar to:

```json
{
  "analysis_id": "VM-XXXXXXXX",
  "media_type": "image",
  "module": "image",
  "status": "success",
  "scores": {},
  "evidence": [],
  "artifacts": [],
  "timestamps": [],
  "warnings": [],
  "model_versions": {}
}
```

Every module should clearly define:

* Score direction
* Model version
* Processing status
* Evidence
* Warnings
* Missing information

Missing metadata must not automatically be treated as evidence of manipulation.

---

## 11. Installation

Clone the repository:

```cmd
git clone https://github.com/ShubhiGupta155/VeriMediaAI.git
cd VeriMediaAI
```

Create a virtual environment:

```cmd
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

---

## 12. Running the Application

The current prototype uses Streamlit.

Run:

```cmd
python -m streamlit run app.py
```

The application will open in your browser.

---

## 13. Example Analysis Workflow

```text
Upload Media
     ↓
Identify Media Type
     ↓
Preprocess Media
     ↓
Run Specialized Detectors
     ↓
Extract Forensic Evidence
     ↓
Analyze Regions / Frames / Timestamps
     ↓
Perform Multimodal Fusion
     ↓
Calculate Risk Assessment
     ↓
Generate DeepSeek Explanation
     ↓
Create Final Forensic Report
```

---

## 14. Screenshots / Demo

Screenshots will be added as the application UI and analysis pipeline are developed.

Planned screenshots:

```text
docs/screenshots/
├── dashboard.png
├── image-analysis.png
├── video-analysis.png
├── audio-analysis.png
└── forensic-report.png
```

---

## 15. Evaluation

The project will evaluate both individual modules and the complete multimodal system.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* False Positive Rate
* False Negative Rate

Where probabilities are reported, calibration will also be evaluated.

### Planned Experiments

1. Single detector vs multiple forensic signals
2. Single modality vs multimodal analysis
3. Visual-only vs Visual + Audio
4. Visual + Audio vs Visual + Audio + Lip-Sync
5. Different fusion strategies
6. Original media vs platform-compressed media

---

## 16. Datasets

Potential datasets include:

### Images

* Defactify
* Other suitable AI-generated image datasets

### Videos

* DeepfakeBench-compatible datasets
* Suitable deepfake video datasets

### Audio

* ASVspoof datasets

Dataset licenses and usage terms will be checked before redistribution or publication.

---

## 17. Scientific Limitation

No AI detector can guarantee perfect detection of every AI-generated or manipulated media file.

Performance may change because of:

* New generative models
* Image compression
* Resizing
* Re-encoding
* Social-media processing
* Unseen generation techniques
* Dataset differences

Therefore, VeriMedia AI provides an:

> **Evidence-based forensic assessment rather than absolute proof of authenticity.**

The system should be considered a **decision-support tool**, not a replacement for professional forensic investigation.

---

## 18. Security

The project follows basic security practices:

* Validate uploaded files
* Restrict allowed file types
* Limit file sizes
* Generate SHA-256 hashes
* Avoid storing API keys in source code
* Use environment variables for secrets
* Prevent arbitrary file execution
* Maintain analysis IDs
* Record model versions

---

## 19. Development Workflow

The project uses Git and GitHub.

### Main Branch

```text
main
```

### Development Branches

```text
member1-image
member2-video
member3-audio-lipsync
member4-backend-fusion
```

### Workflow

```text
Create Branch
      ↓
Development
      ↓
Testing
      ↓
Commit
      ↓
Push
      ↓
Pull Request
      ↓
Code Review
      ↓
Merge into main
```

Direct development on `main` should be avoided.

---

## 20. Current Project Status

### Completed

* [x] Initial project concept
* [x] Git repository initialization
* [x] Initial project structure
* [x] Streamlit prototype
* [x] Initial image detector prototype
* [x] GitHub repository setup

### In Development

* [ ] Image forensic pipeline
* [ ] Image detector benchmarking
* [ ] Video forensic pipeline
* [ ] Audio forensic pipeline
* [ ] Lip-sync pipeline
* [ ] Evidence fusion
* [ ] Backend API
* [ ] Database
* [ ] DeepSeek integration
* [ ] Automated forensic reports
* [ ] Complete evaluation
* [ ] Final deployment

---

## 21. Future Scope

Future versions may include:

* Additional AI-generated image detectors
* Advanced deepfake detection models
* Improved synthetic voice detection
* Robustness testing against compression
* Cross-model generalization experiments
* Explainable AI visualizations
* Advanced multimodal fusion models
* Cloud deployment
* Larger forensic datasets
* Continuous model benchmarking

---

## 22. Team

**Project:** VeriMedia AI

**Project Type:** B.Tech 

**Team Size:** 4 Members

### Project Goal

> Build an evidence-based multimodal system for analyzing the authenticity of modern digital media.

---

## 23. Disclaimer

VeriMedia AI is an academic/research project intended to assist with digital media forensic analysis.

Its results should not be treated as definitive proof of authenticity or manipulation without appropriate human and forensic verification.

---

## License

This project is developed for academic and research purposes.
