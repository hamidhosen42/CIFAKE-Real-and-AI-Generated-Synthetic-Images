# CIFAKE: Explainable Deep Learning for Classifying Real and AI-Generated Images Using CNN and 3D-CNN

## 📘 Abstract
Artificial intelligence has advanced to a point where synthetic images created by GANs and diffusion models are extremely realistic, making real vs fake image detection challenging.  
This paper evaluates two models — **CNN** and **3D-CNN** — on the **CIFAKE dataset** (120,000 balanced images).  
Explainable AI methods such as **LIME** and **Grad-CAM** are used for transparency.

**Results:**  
- CNN → **95.69% accuracy**, **98.00% recall**  
- 3D-CNN → **96.62% accuracy**, **95.97% precision**, **97.33% recall**, **96.64% F1-score**  

Both models perform well, but 3D-CNN achieves superior robustness.

---

# 1. Introduction
Generative models like **GANs**, **StyleGAN**, and **Stable Diffusion** now produce images nearly indistinguishable from real images. This introduces risks in:

- Misinformation  
- Forgery  
- Identity fraud  
- Digital forensics  
- Media manipulation  

The **CIFAKE dataset** (60K real + 60K synthetic images) provides a strong benchmark for developing reliable detection models.

---

# 2. Dataset & Preprocessing

## 📂 Dataset Sample
![Dataset](Image/dataset.png)

### Dataset Split
| Split | REAL | FAKE | Total |
|--------|--------|--------|--------|
| Training | 45,000 | 45,000 | 90,000 |
| Validation | 5,000 | 5,000 | 10,000 |
| Testing | 10,000 | 10,000 | 20,000 |
| **Total** | **60,000** | **60,000** | **120,000** |

---

## Preprocessing Steps  
Each image is transformed into **6 additional processed versions**, including:

- Green channel extraction  
- CLAHE  
- Gaussian blur  
- Grayscale  
- Canny  
- Sobel  

![Preprocess](Image/preprocess.png)  
![Preprocess Dataset](Image/preprocess_dataset.png)

Total after preprocessing:

- **600,000 training images**
- **60,000 validation**
- **120,000 testing**

---

# 3. Methodology

## 3.1 CNN Architecture
![CNN Model](Image/CNN.png)

Architecture layers:

- Conv2D → BatchNorm → ReLU  
- MaxPooling  
- Dropout (0.2 → 0.5)  
- Global Average Pooling  
- Dense + Sigmoid  

---

## 3.2 3D-CNN Architecture
![3D-CNN Model](Image/3DCNN.png)

- 3×3×3 spatiotemporal kernels  
- 5 convolutional blocks  
- Asymmetric MaxPool3D  
- Dropout (0.2 → 0.5)  
- GlobalAvgPool3D  
- Dense + Sigmoid  

---

# 4. Results

## 4.1 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|-----------|---------|-----------|
| CNN | 95.69% | 93.67% | 98.00% | 95.79% |
| **3D-CNN** | **96.62%** | **95.97%** | **97.33%** | **96.64%** |

---

## 4.2 ROC Curve  
![3D ROC](Image/3DROC.png)

---

## 4.3 Confusion Matrix (3D-CNN)
![Confusion Matrix](Image/3DCNN_confusion matrix.png)

---

## 4.4 Grad-CAM Visualisation

### For 3D-CNN:
![GradCAM](Image/3DGrad_CAM.png)
![GradCAM1](Image/3DGrad_CAM1.png)
![GradCAM2](Image/3DGrad_CAM2.png)

### For CNN:
![2DGrad](Image/3dGrad.png)

---

## 4.5 LIME Visualisation
![LIME](Image/lime1.png)
![LIME2](Image/lime2.png)
![3D LIME](Image/3dlime.png)

---

# 5. Comparison With Previous Works

| Method | Accuracy |
|--------|----------|
| Bird et al. | 92.98% |
| Epstein et al. | 99.2% |
| Baraheem et al. | 100% |
| Saskoro et al. | 96% |
| **Our CNN** | **95.69%** |
| **Our 3D-CNN** | **96.62%** |

---

# 6. Conclusion

- 3D-CNN outperforms conventional CNN  
- Preprocessing + augmentation significantly improves robustness  
- XAI (Grad-CAM & LIME) increases transparency  
- Highly effective for synthetic media detection  

Future directions:

- Transformer-based models  
- SHAP explainability  
- Testing on newer diffusion models  
- Lightweight deployment for real-time use  

---

# 📚 Citation

### **BibTeX**
```bibtex
@inproceedings{hosen2025cifake,
  title={CIFAKE: Explainable Deep Learning for Classifying Real and AI-Generated Images Using CNN and 3D-CNN},
  author={Hosen, Md. Hamid and Asif, Mikdad Mohammad and Uddin, Altaf and Chowdhury, Rituparna and Bhottacharjee, Pappuraj and Saha, Arnob},
  booktitle={2025 IEEE International Conference on Biomedical Engineering, Computer and Information Technology for Health (BECITHCON)},
  year={2025},
  organization={IEEE},
  address={Dhaka, Bangladesh}
}


---

## 📬 Contact

For questions or collaborations:  
📧 [mdhamidhosen4@gmail.com](mailto:mdhamidhosen4@gmail.com)

---
