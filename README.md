# 🧬 PathoMozhi: A Flamingo-Style Vision-Language Model for Histopathology

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model on HuggingFace](https://img.shields.io/badge/HuggingFace-oyemainhun/pathoMozhi-yellow.svg)](https://huggingface.co/oyemainhun/pathoMozhi)

> **License**: This repository is licensed under the [MIT License](LICENSE).

---

## 🚀 Overview

**PathoMozhi** is a Flamingo-inspired [4] vision-language model tailored for digital pathology. It integrates a pretrained language model (BioGPT-Large) with visual context from whole-slide image (WSI) features using gated cross-attention layers.  

---

## 🧪 Contributions

### 2.1 Flamingo-style VLM inspired by PRISM and HistoGPT

We implement a vision-language architecture inspired by recent models such as **PRISM** [1] and **HistoGPT** [2]. A pretrained language model (**BioGPT-Large**) [5] is augmented with cross-attention layers to receive context from WSI-derived image features.

### 2.2 Patch-level MIL using CONCHv1.5 features

Rather than using raw image pixels, we extract **patch-level features** using the **CONCHv1.5** [3] encoder. These are processed in a **multiple instance learning (MIL)** setup before being passed to the language model.

### 2.3 Gated cross-attention + decoupled optimization

- **Learnable gates**: We retain the **gated cross-attention** modules (i.e., `attn_gate`, `ff_gate`) from Flamingo. Unlike the original Flamingo implementation, we **initialize `attn_gate` to 0.55**, allowing partial vision-language interaction at the start of training.
- **Custom parameter grouping**: Gated parameters are trained with a **separate learning rate** (`gate_lr`) using a custom optimizer grouping strategy.

---
## 📚 References


## 📦 Installation

```bash
git clone https://github.com/Srividhya-Sainath/pathoMozhi.git
cd pathoMozhi
pip install .
