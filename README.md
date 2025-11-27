# CIFAKE: Explainable Deep Learning for Classifying Real and AI-Generated Images Using CNN and 3D-CNN

## 📘 Abstract
Artificial intelligence has advanced to a point where synthetic images created by GANs and diffusion models are extremely realistic, making real vs fake image detection challenging.  
This paper evaluates two models — **CNN** and **3D‑CNN** — on the **CIFAKE dataset** (120,000 balanced images).  
Explainable AI methods such as **LIME** and **Grad‑CAM** are used for transparency.

**Results:**  
- CNN → **95.69% accuracy**, **98.00% recall**  
- 3D‑CNN → **96.62% accuracy**, **95.97% precision**, **97.33% recall**, **96.64% F1-score**  

Both models perform well, but 3D‑CNN achieves superior robustness.

---

## 1. Introduction
Generative models like **GANs**, **StyleGAN**, and **Stable Diffusion** now produce images nearly indistinguishable from real images. This raises concerns in:

- Misinformation  
- Forgery  
- Identity fraud  
- Digital forensics  
- Media manipulation  

Traditional detection methods fail because synthetic images replicate natural lighting, depth, and texture.

Deep Learning (CNNs, Transformers, Hybrid Models) is now widely used to detect such fakes, requiring large balanced datasets and strong generalization ability.

The **CIFAKE dataset**, consisting of **60k real + 60k synthetic images**, provides a standardized benchmark for evaluating detection systems.

---

## 2. Literature Review
A summarized comparison from prior work:

| Author | Dataset | Model | Accuracy | Limitations |
|--------|---------|--------|----------|-------------|
| Bird et al. | CIFAKE | CNN + XAI | 92.98% | No cross-dataset evaluation |
| Epstein et al. | 570k images | Online model w/ CutMix | 99.2% | Sensitive to generator architecture |
| Baraheem et al. | 24k | VGG, DenseNet, EfficientNet | 100% | Misclassifies GAN images with sharp textures |
| Saskoro et al. | 500k | Gated CNN | 96% | Dependent on dataset diversity |
| Rodriguez et al. | 1,252 | CNN on PRNU/ELA | >95% | Works only on JPEG |
| Our Work | CIFAKE | CNN & 3D‑CNN | **96.62%** | Higher training cost for 3D-CNN |

---

## 3. Methodology

### 3.1 Dataset Description
According to page 2 of the paper:  
fileciteturn1file0

| Split | REAL | FAKE | Total |
|--------|--------|--------|--------|
| Training | 45,000 | 45,000 | 90,000 |
| Validation | 5,000 | 5,000 | 10,000 |
| Testing | 10,000 | 10,000 | 20,000 |
| **Total** | **60,000** | **60,000** | **120,000** |

The dataset originally contained 120k images but was expanded through preprocessing (next section).

---

## 3.2 Image Preprocessing
Each image was transformed into **6 additional versions**:

- Green channel extraction  
- CLAHE  
- Gaussian blur  
- Grayscale  
- Canny edge detection  
- Sobel gradient magnitude  

➡ Total Training Images → **600,000**  
➡ Validation Images → **60,000**  
➡ Test Images → **120,000**

Fig. 1 (page 2) shows sample preprocessing outputs.  
fileciteturn1file0

---

## 3.3 Image Augmentation
Used to reduce overfitting:

- Random rotation (±20°)  
- Horizontal/vertical flip  
- Zoom (80–120%)  
- Translation (10%)  
- Contrast shift (0.2)

Augmentation was applied **per batch**, improving generalization.

---

## 3.4 CNN Model Architecture
(According to Table III, page 3)  
fileciteturn1file0

5 convolutional blocks:

- 2×Conv2D (3×3), BatchNorm, ReLU  
- MaxPooling  
- Dropout (0.2 → 0.5)  
- Filters: 32 → 64 → 128 → 256 → 512  
- Global Average Pooling  
- Dense + Sigmoid  

---

## 3.5 3D‑CNN Architecture
(According to Table IV, page 3)  
fileciteturn1file0

Uses spatiotemporal kernels (3×3×3) to capture variability across stacked image channels.

- 5 blocks of Conv3D + BatchNorm  
- MaxPool3D with asymmetric pooling  
- Dropout (0.2 → 0.5)  
- Global Average Pool3D  
- Dense + Sigmoid  

---

## 3.6 Loss Function & Optimization
Binary Cross-Entropy:

\[
L = - rac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat y_i) + (1-y_i) \log(1-\hat y_i)]
\]

Optimizers:

| Model | Learning Rate |
|-------|----------------|
| CNN | 1e‑5 |
| 3D‑CNN | 1e‑4 |

Early stopping used for best validation loss.

---

## 4. Results

### 4.1 Model Comparison
From Table V (page 4):  
fileciteturn1file0

| Model | Accuracy | Precision | Recall | F1‑Score |
|-------|-----------|-----------|---------|-----------|
| CNN | 95.69% | 93.67% | 98.00% | 95.79% |
| **3D-CNN** | **96.62%** | **95.97%** | **97.33%** | **96.64%** |

➡ **3D‑CNN is the best-performing model**

---

### 4.2 Confusion Matrix (3D‑CNN)
From Table VI:  
fileciteturn1file0

| True / Predicted | FAKE (0) | REAL (1) |
|------------------|----------|----------|
| **FAKE** | 9,591 | 409 |
| **REAL** | 267 | 9,733 |

Misclassification:

- 409 fake images predicted as real  
- 267 real images predicted as fake  

---

### 4.3 Grad‑CAM Visualisation
(See Fig. 3, page 5)  
fileciteturn1file0

Observations:

- REAL images → focused activation on meaningful object areas  
- FAKE images → diffuse activation, irregular textures  
- Model detects **synthetic artifacts**, not object semantics  

---

### 4.4 LIME Visualisation
(See Fig. 4, page 5)  
fileciteturn1file0

- REAL → continuous contours highlighted  
- FAKE → fragmented, inconsistent patches  
- Confirms 3D‑CNN’s robustness in finding anomalies.

---

## 5. Comparison With Previous Works
From Table VII (page 6):  
fileciteturn1file0

| Method | Accuracy |
|--------|----------|
| Bird et al. | 92.98% |
| Epstein et al. (CutMix) | 99.2% |
| Baraheem et al. | 100% |
| Saskoro et al. | 96% |
| **Our CNN** | **95.69%** |
| **Our 3D‑CNN** | **96.62%** |

Our models provide:

- High accuracy  
- Full explainability  
- Balanced real vs fake dataset  
- Stronger generalization than basic CNN baselines  

---

## 6. Conclusion

This research provides a complete deep-learning pipeline using **CNN** and **3D‑CNN** for distinguishing real vs AI-generated images on the CIFAKE dataset.

- Preprocessing & augmentation created robust feature variety  
- CNN achieved strong results  
- **3D‑CNN outperformed all with 96.62% accuracy**  
- XAI techniques (Grad‑CAM & LIME) improved interpretability  

Future improvements:

- Use transformers for deeper global understanding  
- Add SHAP explainability  
- Test on newer diffusion-model outputs  
- Deploy lightweight real-time detectors

---

``
@inproceedings{hosen2025cifake,
  title={CIFAKE: Explainable Deep Learning for Classifying Real and AI-Generated Images Using CNN and 3D-CNN},
  author={Hosen, Md. Hamid and Asif, Mikdad Mohammad and Uddin, Altaf and Chowdhury, Rituparna and Bhottacharjee, Pappuraj and Saha, Arnob},
  booktitle={2025 IEEE International Conference on Biomedical Engineering, Computer and Information Technology for Health (BECITHCON)},
  year={2025},
  organization={IEEE},
  address={Dhaka, Bangladesh}
}
```
