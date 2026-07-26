# sizcal
Sizcal is an AI-powered computer vision tool and browser extension that captures real-time body measurements (neck, chest, waist, inseam) via webcam, matches them to brand sizing charts across all genders, and automatically filters e-commerce storefronts to display accurately fitting clothes.
# 📏 Sizcal – Real-Time Virtual Body Sizing & E-Commerce Filter Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?logo=javascript&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00979D?logo=google&logoColor=white)
![Manifest V3](https://img.shields.io/badge/Chrome_Extension-Manifest_V3-4285F4?logo=googlechrome&logoColor=white)

**Sizcal** bridges the gap between online apparel shopping and physical sizing. By leveraging browser-based computer vision, Sizcal calculates key body measurements using a webcam, guides user positioning in real-time, maps measurements to brand-specific size charts across all genders, and automatically filters target e-commerce platforms to display relevant inventory.

---

## ✨ Key Features

- **📷 In-Browser Pose & Landmark Measurement:** Calculates physical dimensions (neck length, chest/bust, waist, hips, inseam) in real time using edge-computed pose landmark extraction.
- **🎯 Dynamic Camera Guidance:** Provides visual cues ("Step Back", "Adjust Angle", "Hold Pose") to ensure correct camera distance and framing.
- **📐 Scale Calibration Engine:** Converts raw pixel distances to real-world units ($cm$/$in$) using standard reference ratios and height heuristics.
- **🏷️ Multi-Gender Brand Mapper:** Normalizes body measurements against brand size charts (Men's, Women's, Unisex, European, Asian fits) to return confidence-scored size recommendations ($S, M, L, XL$).
- **🛍️ Auto-Filter Extension:** Integrates via Browser Extension (Manifest V3) to select and filter size options directly on active shopping pages.
- **🔒 Privacy-First Design:** All image processing runs locally in the browser. No video frames or photos are uploaded or stored.

---

## 🏗️ System Architecture

```text
+-------------------------------------------------+
|      User Opens E-Commerce Sizing Interface     |
+------------------------+------------------------+
                         |
                         v
+-------------------------------------------------+
|      Camera Calibration & Visual Guidance       |
+------------------------+------------------------+
                         |
                         v
+-------------------------------------------------+
|      In-Browser Inference (MediaPipe / WASM)    |
+------------------------+------------------------+
                         |
                         v
+-------------------------------------------------+
|      Real-World Unit Measurement Calculator     |
+------------------------+------------------------+
                         |
                         v
+-------------------------------------------------+
|      Brand Size Chart Mapper (Multi-Gender)     |
+------------------------+------------------------+
                         |
                         v
+-------------------------------------------------+
|      DOM Injection / Site Catalog Auto-Filter   |
+-------------------------------------------------+

## 🛠️ Tech Stack

- **Core Vision & AI:** Google MediaPipe Pose, TensorFlow.js, OpenCV (WebAssembly)
- **Frontend / Extension:** JavaScript (ES6+), HTML5 Canvas, Chrome Extension Manifest V3
- **Backend Services:** Node.js / FastAPI (Python)
- **Database:** PostgreSQL (Brand size chart mappings)

---

## 🚀 Getting Started

### Prerequisites

- Node.js `v18.0.0+` or Python `3.10+`
- Chromium-based browser (Chrome, Edge, Brave) with webcam permissions

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/sizcal.git](https://github.com/your-username/sizcal.git)
   cd sizcal

🛡️ Privacy
Sizcal executes all image processing client-side. Webcams are active strictly during the measurement modal state, and raw video frames never leave the client device.
