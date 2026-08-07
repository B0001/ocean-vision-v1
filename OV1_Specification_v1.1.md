# Functional Engineering Specification: Ocean Vision (OV-1)

**Document Version:** 1.1
**Target Domain:** Open-Water Human Distress & Dynamic Ocean Anomaly Detection
**Status:** DRAFT

## 1. System Overview & Objective

The **OV-1 System** is an edge-deployed, real-time spatiotemporal computer vision pipeline designed to monitor open-water environments. It is engineered to operate in the chaotic thermodynamic environment of the open ocean, explicitly rejecting naive spatial classification models in favor of physics-informed temporal anomaly detection.

Its primary objectives are:

1.  **Human Distress Detection:** Identify active and passive drowning dynamics in non-linear, high-turbidity wave fields, discriminating between swimming, distress, and inanimate debris.
2.  **Hydrographic Anomaly Identification:** Detect macroscopic oceanographic threats, specifically rip currents, sudden drop-offs, and localized turbulence spikes.
3.  **Deterministic Alerting:** Trigger deterministic, sub-second alerts and provide geospatial tracking data to localized safety response teams without reliance on fragile cloud infrastructure.

## 2. Technical Performance Requirements

| Metric | Target Boundary | Failure Threshold | Rationale |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | $\le 250	ext{ ms}$ | $> 500	ext{ ms}$ | Sub-second response required for effective intervention. |
| **Temporal Window** | 120 frames @ 30 FPS | $< 60	ext{ frames}$ | Minimum baseline required to capture the full frequency of active drowning responses. |
| **False Negative Rate** | $< 0.01\%$ | $\ge 0.05\%$ | A false negative is a critical failure resulting in potential loss of life. |
| **False Positive Rate** | $< 2.0\%$ | $\ge 5.0\%$ | High FPR leads to alarm fatigue among response teams. |
| **Ingress Resolution** | 4K @ 60 FPS | $< 1080	ext{p}$ | Required for tracking sub-pixel motion vectors at extended ranges. |
| **Power Envelope** | $\le 150	ext{W}$ per node | $> 250	ext{W}$ | Constraints of solar/battery powered remote edge installations. |

## 3. System Architecture & Data Pipeline

The system utilizes a hybrid architecture, processing multimodal sensor inputs through a physics-informed preprocessing layer before feeding into a spatiotemporal deep learning pipeline.

```mermaid
graph TD
    A[Dual-Sensor Array: Optical + Polarized NIR] --> B(Hardware Preprocessing Engine)
    B -->|Specular Glare Masking| C{Color Space Normalization & Attenuation}
    C --> D[Spatiotemporal Feature Extractor]
    D -->|Sparse Optical Flow v[x,y,t]| E[Dual-Head Inference Model]
    D -->|3D-CNN / ViT Embeddings| E
    E --> F[Head A: Swimmer Trajectory Autoencoder]
    E --> G[Head B: Hydrodynamic Anomaly Classifier]
    F -->|Anomaly Score > Threshold| H(Deterministic Alerting Logic Engine)
    G -->|Rip Current / Threat Detected| H
    H --> I[Local Alarm Relay & GPS Broadcast]
```

## 4. Subsystem Specifications

### 4.1 Ingress & Optical Preprocessing Module

The primary challenge in ocean vision is the dynamic optical properties of water.

*   **Multimodal Sensors:** Co-axial dual-lens sensors (RGB Optical + Polarized Near-IR 850nm).
*   **Physics-Based Glare Suppression:** Real-time computation of linear degree of polarization (DoLP) based on Stokes parameters. This dynamically masks specular solar reflections from moving wave crests prior to tensor generation, a critical step for reducing high-frequency optical noise.
*   **Depth Attenuation Normalization:** Standard RGB color spaces fail due to rapid attenuation of long wavelengths ($e^{-\eta d}$). The pipeline mandates conversion to **Lab color space**. Dynamic contrast stretching is applied strictly to the $a^*$ and $b^*$ chromaticity channels to mathematically compensate for depth-dependent light absorption.

### 4.2 Spatiotemporal Inference Pipeline

Static object detection (e.g., standard YOLO) is explicitly banned from the distress detection pipeline.

*   **Architecture:** Hybrid 3D Convolutional Neural Network (Conv3D) integrated with Sparse Optical Flow Vector estimation.
*   **Input Tensor Specification:** $(B, C, T, H, W) = (1, 4, 120, 1080, 1920)$ where the feature channels $C$ comprise $[	ext{Luminance}, a^*, b^*, 	ext{DoLP}]$.
*   **Unsupervised Anomaly Detection:** The primary detection mechanism is a Spatiotemporal Autoencoder trained *exclusively* on baseline, non-distress aquatic motion patterns.
*   **Trigger Condition:** 
    Let $X$ be the input sequence and $\hat{X}$ be the reconstructed sequence. 
    The Reconstruction Error Score is calculated as $S_r = ||X - \hat{X}||^2$.
    An alert is triggered if $S_r > 	au_{drowning}$ continuously over a sliding temporal window $\Delta t \ge 3.0	ext{s}$. This detects the high-frequency, low-displacement thrashing characteristic of active drowning.

### 4.3 Hydrodynamic Anomaly Module

This secondary head monitors the environment for structural hazards.

*   **Rip Current Identification Logic:** The system tracks the spatial movement of dense foam tracer particles (generated by breaking waves). If the continuous localized velocity field $ec{v}(x,y,t)$ points seaward with a sustained magnitude $|ec{v}| > 0.5	ext{ m/s}$ across a spatial channel width of less than $15	ext{m}$, the region is classified as an active Rip Current hazard.

## 5. Deployment & Hardware Constraints

1.  **Strict On-Premise Edge Compute:** Inference must execute locally on ruggedized, salt-fog resistant edge hardware (e.g., NVIDIA Jetson AGX Orin industrial arrays).
2.  **Zero-Dependency Fail-Safe:** The detection and alerting engine **must not require internet connectivity** to function. Cloud dependencies for life-safety systems are unacceptable due to latency and connection fragility in coastal zones. Network connectivity is used *only* for asynchronous telemetry and model weight updates.
3.  **Environmental Tolerance:** Sensor housings must meet IP68 ratings and include active thermal management to operate in direct solar load conditions.
