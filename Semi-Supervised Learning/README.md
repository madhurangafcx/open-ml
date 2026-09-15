# Semi-Supervised Learning

**Semi-Supervised Learning (SSL)** operates at the intersection of supervised and unsupervised learning. In many practical scenarios, acquiring ground-truth annotations requires expensive human labor or domain experts, whereas raw, unlabeled samples are abundant.

SSL leverages a small amount of labeled data $(X_L, y_L)$ alongside a large reservoir of unlabeled data $(X_U)$ to build models that achieve higher predictive accuracy than could be produced using labeled data alone.

$$\mathcal{D} = \{(x_1, y_1), \dots, (x_l, y_l)\} \cup \{x_{l+1}, \dots, x_{l+u}\} \quad \text{where } l \ll u$$

---

## Core Problem Paradigms & Key Methods

### 1. Self-Training & Pseudo-Labeling
A wrapper-based iterative strategy where a base classifier is initially trained on labeled data $(X_L, y_L)$.
* **Pseudo-Labeling**: The model generates predictions for the unlabeled set $X_U$. Unlabeled samples whose predicted probability exceeds a predefined confidence threshold $\tau$ are assigned "pseudo-labels".
* These pseudo-labeled instances are merged back into the training dataset, and the model is iteratively retrained to refine its decision boundary.
* **Co-Training**: Multiple models are trained on distinct, conditionally independent feature views of the data, mutually teaching and pseudo-labeling data for one another.

### 2. Graph-Based Methods
Models that represent data points as nodes in a graph, with edge weights reflecting sample similarity (e.g., Euclidean distance or Gaussian RBF kernel).
* **Label Propagation**: Propagates known labels through the graph via continuous Markov random walks or harmonic energy minimization until convergence, assigning labels based on community structure.
* **Label Spreading**: A variation of label propagation that introduces regularization to handle noisy ground-truth labels and prevent overfitting.

### 3. Generative Models (SGANs)
* **Semi-Supervised GANs (SGANs)**: An extension of Generative Adversarial Networks where the Discriminator is converted into an active classifier. Instead of making a binary decision (real vs. fake), the discriminator outputs a distribution over $K + 1$ classes (the $K$ true labeled classes plus a $(K+1)$-th class representing generated fake data). This forces the network to learn rich semantic representations from unlabeled real images.

### 4. Consistency Regularization & Modern Deep SSL
Based on the smoothness assumption: small perturbations to an input should not significantly change the model's output distribution.
* **Mean Teacher**: Employs an exponential moving average (EMA) of student model weights to form a "teacher" network, enforcing consistency between teacher and student outputs under stochastic augmentations.
* **MixMatch & FixMatch**: State-of-the-art frameworks combining strong and weak data augmentations with pseudo-labeling and consistency loss to achieve near-supervised accuracy with fewer than 1% labeled samples.

---

## Roadmap & Planned Notebooks

| Topic | Planned Directory | Description | Status |
| :--- | :--- | :--- | :--- |
| **Pseudo-Labeling / Self-Training** | `Pseudo_Labeling/` | Baseline self-training on vision/tabular data | :black_square_button: Planned |
| **Label Propagation** | `Label_Propagation/` | Graph-based transductive classification | :black_square_button: Planned |
| **Semi-Supervised GANs (SGAN)** | `SGAN/` | PyTorch SGAN on benchmark vision datasets | :black_square_button: Planned |
| **Consistency Regularization (FixMatch)** | `FixMatch/` | Weak-to-strong augmentation consistency loss | :black_square_button: Planned |
