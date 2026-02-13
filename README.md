# [AI-Image-Authenticity-Analyzer]()
Multi-task Vision Transformer system for detecting AI-generated images and identifying their source models.

CPU-optimized **AI Image Authenticity Analyzer** (Streamlit) powered by a **Multi-Task Vision Transformer (ViT)** trained on Defactify.  
It predicts **(A) Real vs AI-Generated** and **(B) Generator source** (SD21/SDXL/SD3/DALL·E 3/Midjourney).

> Disclaimer: This tool is **probabilistic**. Use it as guidance, not as legal/forensic proof.

---

## Features

- **Task A:** Real vs AI-Generated (P(AI) + threshold-based verdict)
- **Task B:** Source generator prediction (Top-K)
- **Explanation Heatmap (Occlusion):**
  - Explain **P(AI)** or **Decision confidence** or **Generator confidence**
  - CPU-safe with **time limit** and caching
- **Robustness Check:**
  - JPEG compression variants + resize stress test
  - Outputs variance + flip-rate + robustness score
- **Streamlit-ready UI**
  - Checkpoint **dropdown** auto-selects `best*` if present
  - Works on **CPU-only** hosting (Streamlit Community Cloud)

---

## Dataset (Biggest Credit) 

Training uses the [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) from Hugging Face:  
