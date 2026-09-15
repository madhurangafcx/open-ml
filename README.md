# Open ML

A structured, hands-on repository covering foundational algorithms, lab exercises, and modern paradigms in Machine Learning and Deep Learning.

---

## Machine Learning Taxonomy

| Learning Paradigm | Overview & Objectives | Key Algorithms & Methods | Section README |
| :--- | :--- | :--- | :---: |
| **Supervised Learning** | Learns mapping from inputs ($X$) to labels ($y$) using ground truth pairs. | Linear Regression, Logistic Regression, Decision Trees, Random Forests, SVM, KNN, Naive Bayes, ANNs, CNNs, RNNs/LSTMs, Transformers | [Explore](./Supervised%20Learning/README.md) |
| **Unsupervised Learning** | Identifies patterns, groupings, and low-dimensional manifolds in unlabeled data. | K-Means, Hierarchical Clustering, DBSCAN, GMM, PCA, t-SNE, UMAP, Autoencoders, Apriori | [Explore](./Unsupervised%20Learning/README.md) |
| **Reinforcement Learning** | Learns optimal policies via agent-environment interactions and reward feedback. | Q-Learning, SARSA, Deep Q-Networks (DQN), Policy Gradient, PPO, Actor-Critic (A2C/A3C), DDPG, SAC | [Explore](./Reinforcement%20Learning/README.md) |
| **Semi-Supervised Learning**| Combines small labeled datasets with abundant unlabeled data to improve accuracy. | Self-Training / Pseudo-Labeling, Co-Training, Label Propagation, Consistency Regularization, Mean Teacher, SGANs, FixMatch | [Explore](./Semi-Supervised%20Learning/README.md) |

---

## Repository Structure & Progress

```text
.
├── Readme.md                          # Root repository overview & taxonomy
├── Supervised Learning/               # Labeled learning (Regression & Classification)
│   ├── README.md                      # Detailed paradigm overview & roadmap
├── Unsupervised Learning/             # Unlabeled pattern discovery & clustering
│   └── README.md                      # Clustering, dimensionality reduction & association rules
├── Reinforcement Learning/            # Trial-and-error agent learning & rewards
│   └── README.md                      # Value-based, policy gradient & actor-critic
└── Semi-Supervised Learning/          # Hybrid labeled/unlabeled learning
    └── README.md                      # Pseudo-labeling, graph methods, SGANs & FixMatch
```

---

## Getting Started

### Prerequisites
* Python 3.9+
* Recommended environment manager: `conda` or Python `venv`

### Installation
Clone or download the repository, then install standard machine learning libraries:

```bash
# Clone the repository
git clone https://github.com/<your-username>/open-ml.git
cd open-ml

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install essential packages
pip install numpy pandas scikit-learn matplotlib seaborn jupyter torch torchvision
```

### Running the Notebooks
Launch Jupyter Notebook or JupyterLab from the project root:

```bash
jupyter notebook
```
Navigate to any target folder (such as [`Supervised Learning/Linear_regression`](./Supervised%20Learning/Linear_regression/)) and open the `.ipynb` file.

---

### Note
```text
Please note that some parts of the README might not display correctly due to limited Markdown support or formatting errors in the math syntax.
```

### Author
Created by **Pasan Madhuranga** ([@madhurangafcx](https://github.com/madhurangafcx)) as an open educational foundation for machine learning engineers and researchers.

