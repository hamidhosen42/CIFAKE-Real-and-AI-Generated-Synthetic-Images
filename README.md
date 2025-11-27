# CIFAKE: Explainable Deep Learning for Classifying Real and AI-Generated Images Using CNN and 3D-CNN

**Authors:**  
Md. Hamid Hosen¹, Mikdad Mohammad Asif², Altaf Uddin³, Rituparna Chowdhury⁴,  
Pappuraj Bhottacharjee⁵, Arnob Saha⁶  

¹²³⁴⁶ Department of Computer Science & Engineering, East Delta University, Chittagong, Bangladesh  
⁵ Department of Computer Science & Engineering, Dhaka University of Engineering & Technology, Gazipur, Bangladesh  

📧 mdhamidhosen4@gmail.com, asifmikdad@gmail.com, altafuddinsanim@gmail.com,  
rituparna.chy550@gmail.com, pappuraj.duet@gmail.com, sahaarnob73@gmail.com  

---

## 📘 Abstract

Artificial intelligence has reached a level where distinguishing real images from synthetic ones is increasingly difficult. This creates risks in areas such as misinformation, digital security, and content authenticity. In this study, we address this problem by applying deep learning to the CIFAKE dataset, which contains a balanced collection of real and AI- generated images. Two models were developed: a Convolutional Neural Network (CNN) for extracting spatial features, and a three-dimensional CNN (3D-CNN) for capturing spatiotemporal patterns. The dataset was processed with augmentation and preprocessing to improve model generalization. Both models achieved strong results. The CNN obtained 95.69% accuracy, 93.67% precision, 98.00% recall, and an F1-score of 95.79%. The 3D-CNN outperformed it, achieving 96.62% accuracy, 95.97% precision, 97.33% recall, and an F1-score of 96.64%. To improve interpretability, explainable AI methods were applied. LIME provided local feature explanations, while Grad-CAM produced visual heatmaps of the most influential regions in the images. Together, these methods not only improved detection accuracy but also added transparency to the decision-making process. The results highlight the importance of combining robust classification models with explainable techniques for reliable detection of AI- generated images.

Two models were developed:

- a **Convolutional Neural Network (CNN)** for extracting spatial features  
- a **three-dimensional CNN (3D-CNN)** for capturing spatiotemporal patterns  

The dataset was processed with augmentation and preprocessing to improve model generalization.

**Performance:**

- **CNN:** 95.69% accuracy, 93.67% precision, 98.00% recall, 95.79% F1-score  
- **3D-CNN:** 96.62% accuracy, 95.97% precision, 97.33% recall, 96.64% F1-score  

To improve interpretability, explainable AI methods were applied:

- **LIME** provided local feature explanations  
- **Grad-CAM** produced visual heatmaps of the most influential regions  

The results highlight the importance of combining robust classification models with explainable techniques for reliable detection of AI-generated images.

---

## 🔑 Keywords

**CNN, 3D-CNN, CIFAKE, image classification, explainable AI, LIME, Grad-CAM**

---

## 1. Introduction

Artificial Intelligence (AI) has significantly changed the way images are created, making it increasingly difficult to distinguish real photos from machine-generated ones. Traditional camera images usually contain natural visual cues such as consistent lighting, realistic textures, and proper spatial depth. With the rise of GANs and transformer-driven image generators, synthetic visuals have reached a level of realism where they often appear indistinguishable from genuine photographs.

While this enables new applications in digital design, simulations, and media production, it also raises concerns about the authenticity of visual content and the potential for **misinformation, deepfakes, and identity fraud**.

Traditional detection methods struggle because AI-generated images replicate real textures, lighting, and structural details so well that pixel-level or handcrafted-feature approaches often fail. Deep learning models—especially **CNNs**, transformer-based architectures, and hybrid systems—have therefore become the main tool for detecting AI-generated images.

However, several challenges remain:

- Rapid improvement of generative models  
- Lack of standardized, balanced datasets  
- Limited generalization across different generators  
- Limited interpretability of black-box deep models  

The **CIFAKE dataset**, consisting of 60k real and 60k synthetic images, provides a balanced benchmark for investigating these issues. In this work we:

1. Train and evaluate **CNN** and **3D-CNN** models on CIFAKE.  
2. Apply **preprocessing** and **augmentation** to expand and diversify the dataset.  
3. Use **LIME** and **Grad-CAM** to interpret model decisions.  

**Main contributions:**

- A systematic evaluation of CNN and 3D-CNN for real vs AI image detection on CIFAKE.  
- An extensive preprocessing pipeline where each original image is transformed into six additional views, improving feature diversity.  
- Demonstration that 3D-CNN achieves superior performance (96.62% accuracy, 96.64% F1).  
- Integration of XAI (LIME + Grad-CAM) to provide meaningful visual explanations of model decisions.

---

## 2. Literature Review

The widespread use of AI-generated images threatens the reliability and trustworthiness of media. Accurate detection is needed to prevent misleading data, identify deepfakes, and ensure authenticity.

A brief overview of key related work:

| Author                          | Dataset / Setting                                           | Model(s)                                   | Accuracy            | Key Limitation |
|---------------------------------|-------------------------------------------------------------|--------------------------------------------|---------------------|----------------|
| **Bird et al.** (CIFAKE)       | 120k images (60k real, 60k synthetic)                      | CNN + XAI                                  | 92.98%              | No cross-dataset evaluation |
| **Epstein et al.**             | 570,221 images from 14 generative methods                  | Online learning + CutMix                   | 99–99.2%            | Sensitive to large architecture changes in generators |
| **Baraheem et al.**            | 24,000 images                                              | VGG, ResNet, Inception, DenseNet, EfficientNet | 100% (EfficientNetB4 on their RSI set) | May misclassify sharp-texture GAN outputs |
| **Saskoro et al.**             | 500k images (natural vs AI-generated)                      | Gated CNN + ResNet-50                      | >96%                | Performance highly dependent on training data variety |
| **Martin-Rodriguez et al.**    | 1,252 images (balanced)                                    | CNNs on PRNU/ELA features                  | >95%                | Only applicable to JPEG due to PRNU/ELA constraints |

These works show strong performance but often:

- depend on specific data distributions,
- are tied to particular image formats or generators, or  
- lack interpretability.

Our work aims to provide **balanced performance + strong explainability** on a widely used, publicly available dataset.

---

## 3. Methodology

We evaluate two deep-learning models—**CNN** and **3D-CNN**—on the CIFAKE dataset. The pipeline consists of:

1. Dataset preparation and splitting  
2. Image preprocessing  
3. Data augmentation  
4. Model design (CNN and 3D-CNN)  
5. Training and evaluation  
6. Explainable AI analysis

---

### 3.1 Dataset Description

We use the **CIFAKE** dataset from Kaggle, containing balanced classes:

- **REAL** – natural images  
- **FAKE** – AI-generated images  

| Subset      | REAL  | FAKE  | Total  |
|------------|-------|-------|--------|
| Training   | 45,000 | 45,000 | 90,000 |
| Validation | 5,000  | 5,000  | 10,000 |
| Testing    | 10,000 | 10,000 | 20,000 |
| **Total**  | **60,000** | **60,000** | **120,000** |

> Original split from Kaggle: 100k training, 20k test.  
> We re-split into train/val/test as above.

---

### 3.2 Image Preprocessing

Each image is transformed into **six additional processed versions**:

1. Green-channel extraction  
2. CLAHE on the green channel  
3. Gaussian blur  
4. Grayscale conversion  
5. Canny edge detection  
6. Sobel gradient magnitude  

This yields:

- **600,000 training images** (300k REAL, 300k FAKE)  
- **60,000 validation images** (30k REAL, 30k FAKE)  
- **120,000 test images** (60k REAL, 60k FAKE)

All subsets remain **balanced**.

Example preprocessing outputs:

![Preprocessing examples](Image/preprocess.png)

---

### 3.3 Data Augmentation

To reduce overfitting and improve generalization, we apply online augmentation during training:

- Random rotation up to ±20°  
- Horizontal and vertical flips  
- Random zoom (80–120%)  
- Random translation (up to 10% in both axes)  
- Random contrast adjustment (±0.2)

Augmentation is applied **per batch**, so the model sees different variants of each image across epochs.

---

### 3.4 CNN Architecture

The 2D CNN is designed to capture spatial features from individual images.

| Block | Filters | Kernel | Pooling       | Dropout |
|-------|---------|--------|---------------|---------|
| 1     | 32      | 3×3    | MaxPool 2×2   | 0.2     |
| 2     | 64      | 3×3    | MaxPool 2×2   | 0.3     |
| 3     | 128     | 3×3    | MaxPool 2×2   | 0.3     |
| 4     | 256     | 3×3    | MaxPool 2×2   | 0.4     |
| 5     | 512     | 3×3    | GlobalAvgPool | 0.5     |

Each block consists of:

- 2 × Conv2D → BatchNorm → ReLU  
- MaxPooling  
- Dropout  

Then:

- Global Average Pooling  
- Dense layer(s)  
- Final **sigmoid** neuron for binary classification.

---

### 3.5 3D-CNN Architecture

The 3D-CNN extends the same idea to **3D convolutions** over stacked inputs, capturing spatiotemporal patterns.

| Block | Filters | Kernel      | Pooling          | Dropout |
|-------|---------|-------------|------------------|---------|
| 1     | 32      | 3×3×3       | MaxPool3D (1,2,2) | 0.2     |
| 2     | 64      | 3×3×3       | MaxPool3D (2,2,1) | 0.3     |
| 3     | 128     | 3×3×3       | MaxPool3D (2,2,1) | 0.3     |
| 4     | 256     | 3×3×3       | MaxPool3D (2,2,1) | 0.4     |
| 5     | 512     | 3×3×3       | GlobalAvgPool3D  | 0.5     |

Forward pass of a 2D conv layer:

\[
F_{i,j,k} = \sigma \left( \sum_{m=1}^{M} \sum_{p=1}^{P} \sum_{q=1}^{Q} W_{p,q,m,k} \cdot X_{i+p, j+q, m} + b_k \right)
\]

In 3D-CNN this extends over temporal depth, letting the model learn both spatial and sequential structure.

Training loss: **Binary Cross-Entropy**

\[
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
\]

- Optimizer: **Adam**  
  - CNN LR = 1e-5  
  - 3D-CNN LR = 1e-4  
- Early stopping with best-weight restoration.  

---

### 3.6 Explainable AI (XAI)

We use two model-agnostic XAI methods:

1. **LIME** (Local Interpretable Model-agnostic Explanations)  
   - Perturbs the input image  
   - Learns a local interpretable model of the prediction  
   - Highlights superpixels most responsible for the decision  

2. **Grad-CAM** (Gradient-weighted Class Activation Mapping)  
   - Uses gradients of the target class flowing into the final conv layer  
   - Produces class-specific heatmaps  
   - Highlights spatial regions most influencing the prediction  

These methods allow us to visually inspect **why** the CNN and 3D-CNN classify an image as REAL or FAKE.

---

## 4. Result Analysis

After training, both models were evaluated on the test set. We analyze:

- Training behavior (accuracy & loss)  
- Test-set metrics  
- Confusion matrix  
- XAI visualizations  
- Comparison with previous work  

---

### 4.1 Training and Evaluation

The **3D-CNN** converged faster than the CNN:

- Early-stopped at **epoch 44** (vs 126 for CNN).  
- Validation accuracy stabilized around **96%**.  
- Precision and recall stayed consistently above **95%**.

Training/validation curves for 3D-CNN:

![3D-CNN Training Curves](Image/3DCNN.png)

---

### 4.2 Performance Metrics

| Model  | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| CNN    | 95.69%   | 93.67%    | 98.00% | 95.79%   |
| **3D-CNN** | **96.62%** | **95.97%** | **97.33%** | **96.64%** |

The 3D-CNN shows **more balanced** precision & recall and overall higher F1-score.

---

### 4.3 Confusion Matrix (3D-CNN)

| True \ Predicted | FAKE (0) | REAL (1) |
|------------------|----------|----------|
| **FAKE (0)**     | 9,591    | 409      |
| **REAL (1)**     | 267      | 9,733    |

- 409 FAKE images misclassified as REAL  
- 267 REAL images misclassified as FAKE  

Overall, error rates are low and balanced across both classes.

---

### 4.4 Grad-CAM Visualisation (3D-CNN)

![3D-CNN Grad-CAM](Image/3dGrad.png)

Observations:

- **REAL** images: heatmaps focus on semantically meaningful regions (e.g., body of the dog, outline of a bird).  
- **FAKE** images: activation appears more scattered and less structurally coherent, capturing irregular textures typical of synthetic content.

This confirms that the 3D-CNN leverages object structure for REAL images and artifact-like regions for FAKE images.

---

### 4.5 LIME Visualisation (3D-CNN)

![3D-CNN LIME](Image/3dlime.png)

- For correctly classified REAL images (confidence ~0.98), LIME highlights object contours and meaningful regions.  
- For correctly classified FAKE images, highlighted regions appear fragmented and irregular, reflecting unstable textures.

Together, LIME and Grad-CAM offer strong evidence that the model is learning **useful and interpretable cues** for classification.

---

### 4.6 Comparison With Previous Work

| Work                         | Dataset / Setting                             | Best Metric             |
|-----------------------------|-----------------------------------------------|-------------------------|
| Bird et al. (CIFAKE)        | CIFAKE                                       | 92.98% accuracy         |
| Epstein et al.              | 570k images, 14 generative methods           | 99–99.2% detection      |
| Baraheem et al.             | RSI dataset, 24k images                      | 100% (EfficientNetB4)   |
| Saskoro et al.              | 500k real vs synthetic                       | 96% (Gated CNN)         |
| Martin-Rodriguez et al.     | 1,252 images                                 | >95% accuracy           |
| **Our CNN (CIFAKE)**        | CIFAKE                                       | 95.69% acc, 95.79% F1   |
| **Our 3D-CNN (CIFAKE)**     | CIFAKE                                       | **96.62% acc, 96.64% F1** |

Our models combine **high accuracy** with **strong explainability**, making them practical for real-world AI-image detection scenarios.

---

## 5. Conclusion and Future Work

This study addressed the challenge of distinguishing REAL from AI-generated images using the CIFAKE dataset. We proposed:

- A **CNN** for spatial feature extraction  
- A **3D-CNN** for spatiotemporal pattern learning  

Preprocessing and augmentation expanded each image into multiple transformed variants, greatly increasing feature diversity and improving generalization.

Both models performed strongly, but **3D-CNN** achieved the best results:

- 96.62% accuracy  
- 95.97% precision  
- 97.33% recall  
- 96.64% F1-score  

Explainable AI techniques (LIME and Grad-CAM) revealed that:

- REAL images are classified using coherent object-centric regions.  
- FAKE images are classified using irregular textures and artifact-like regions.  

**Future work:**

- Evaluate transformer-based and hybrid architectures.  
- Incorporate SHAP and other XAI methods for deeper interpretability.  
- Test on newer diffusion/transformer-based generators (e.g., SDXL, DALLE-3, Midjourney).  
- Develop lightweight models suitable for real-time deployment in journalism, forensics, and content verification pipelines.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{hosen2025cifake,
  title={CIFAKE: Explainable Deep Learning for Classifying Real and AI-Generated Images Using CNN and 3D-CNN},
  author={Hosen, Md. Hamid and Asif, Mikdad Mohammad and Uddin, Altaf and Chowdhury, Rituparna and Bhottacharjee, Pappuraj and Saha, Arnob},
  booktitle={2025 IEEE International Conference on Biomedical Engineering, Computer and Information Technology for Health (BECITHCON)},
  year={2025},
  organization={IEEE},
  address={Dhaka, Bangladesh}
}
```

---

## 📬 Contact

For questions or collaborations:  
📧 [mdhamidhosen4@gmail.com](mailto:mdhamidhosen4@gmail.com)

---
