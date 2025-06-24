# Travaux pratiques en Machine Learning & Deep Learning

Ce dépôt regroupe l’ensemble des **travaux pratiques réalisés dans le cadre des cours** suivants de la formation Master en DataScience à la Haute École de Gestion de Genève (HEG) :

Les cours ont été dispensés par le professeur HES **Alexandros Kalousis**, 
la collaboratrice scientifique HES **Frantzeska Lavda** et 
l'assisstant HES et PhD student **Joao Candido Ramos** (qui a principalement supervisé
les labos et les différents fournis pour les TPs).

- 🧠 **Machine Learning**
- 🧬 **Advanced Neural Networks (Deep Learning)**

Chaque TP est accompagné de :
- 🧾 Un **rapport PDF**
- 🧪 Un **notebook Jupyter** ou un script Python pour l'implémentation
- 📊 Des **visualisations** et résultats

---

## 📁 Structure du projet

```bash
.
├── Machine_Learning
│   ├── Naive_Bayes
│   ├── Perceptron_Logistic_Softmax_regression
│   └── QR_Decomposition
│
└── Deep_Learning
    ├── C_VAE
    ├── C_VAEGAN
    ├── Computational_Graph
    └── Convolutional_Neural_Networks
```

---

## 📘 Contenu par dossier

### 🔷 Machine_Learning/

| Dossier | Contenu |
|--------|---------|
| `Naive_Bayes` | Implémentation d’un classifieur bayésien naïf |
| `Perceptron_Logistic_Softmax_regression` | TP complet sur la régression logistique, softmax et perceptron |
| `QR_Decomposition` | Implémentation et application de la décomposition QR |

---

### 🔶 Deep_Learning/

| Dossier | Contenu |
|--------|---------|
| `C_VAE` | TP sur les autoencodeurs variationnels conditionnels |
| `C_VAEGAN` | TP sur la combinaison VAE + GAN |
| `Computational_Graph` | Construction manuelle d’un graphe de calcul pour backpropagation |
| `Convolutional_Neural_Networks` | TP CNN avec PyTorch et dataset CIFAR-10 |

---

## 🛠️ Environnement

Les travaux pratiques utilisent majoritairement :

- `Python 3.9+`
- `NumPy`, `Matplotlib`
- `PyTorch` pour les TPs Deep Learning

Un fichier `requirements.txt` sera ajouté si besoin pour reproduire l’environnement.

---

## 📄 Remarques

- Tous les codes ont été réalisés dans un **but pédagogique**, dans le cadre d’un cursus académique.
- Les données utilisées (ex. CIFAR-10) ne sont **pas versionnées** si elles dépassent les limites de GitHub.
- Les fichiers `.tar.gz`, `.ipynb_checkpoints` ou tout dataset massif sont ignorés via `.gitignore`.

---

## Auteurs

Travaux réalisés par moi (Andi Ramiqi) dans le cadre de la formation Master Data Science à la HEG Genève.
