# Rapport de Projet : Classification de Signaux Radio par Deep Learning

**Auteurs** : Iliass LASRI et Marine VIEILLARD

## 1. Introduction

Ce projet porte sur la classification automatique de signaux radio. L'objectif est de classifier 6 types de modulations différentes (BPSK, QPSK, 8PSK, 16QAM, 64QAM, GMSK) à partir de leurs représentations I/Q (In-phase/Quadrature).

La difficulté majeure réside dans la variabilité du rapport signal sur bruit (SNR), qui varie de 0 dB (signal très bruité) à 30 dB (signal clair). Les performances du modèle doivent donc être robustes à ces variations.

## 2. Données

### 2.1 Structure des données
- **Ensemble d'entraînement** : 30200 échantillons (train.hdf5 + samples.hdf5)
- **Ensemble de validation** : validation.hdf5
- **Ensemble de test** : test.hdf5
- **Format** : Signaux I/Q de longueur 2048, représentés en 2 canaux (I et Q)
- **SNR** : 4 valeurs distinctes (0, 10, 20, 30 dB)
- **Classes** : 6 types de modulation

### 2.2 Augmentation des données
Pour améliorer la généralisation, nous avons implémenté deux techniques d'augmentation :

1. **Rotation I/Q aléatoire** : Rotation aléatoire du signal dans l'espace I/Q (angle entre 0 et 2π)
   - Simule les variations de phase inhérentes aux transmissions radio
   - Appliquée avec probabilité 100%

2. **Dégradation SNR adaptative** : Ajout de bruit pour simuler des conditions plus difficiles
   - SNR 30 dB → peut être dégradé en 20, 10 ou 0 dB
   - SNR 20 dB → peut être dégradé en 10 ou 0 dB
   - SNR 10 dB → peut être dégradé en 0 dB
   - Appliquée avec probabilité 50%

**Impact de l'augmentation** :
- Accélération de la convergence (1/3 du temps d'entraînement)
- Amélioration de l'accuracy sur le test set
- Meilleure robustesse aux variations de SNR

## 3. Architecture des Modèles

### 3.1 Modèle CNN-LSTM avec SNR (Architecture finale)

```
Input: [Batch, 2, 2048] (signaux I/Q)
SNR: [Batch, 1] (valeur du SNR)

├─ Conv1D Block 1: 2 → 32 channels (kernel=7, stride=2)
│  └─ BatchNorm + ReLU
│
├─ Conv1D Block 2: 32 → 64 channels (kernel=5, stride=2)
│  └─ BatchNorm + ReLU
│
├─ LSTM bidirectionnel: 64 → 64*2 hidden units
│  └─ Mean pooling sur dimension temporelle
│
├─ SNR Embedding: SNR → 16 features
│
├─ Concatenation: [LSTM features (128) + SNR features (16)]
│
├─ FC1: 144 → 128 + ReLU
│
└─ FC2: 128 → 6 classes (output logits)
```

**Caractéristiques principales** :
- Extraction de features par CNN 1D
- Modélisation des dépendances temporelles par LSTM bidirectionnel
- Intégration du SNR comme information contextuelle
- ~200K paramètres

### 3.2 Modèles alternatifs explorés

#### ResNet 1D
- Blocs résiduels avec skip connections
- Architecture plus profonde (4 layers avec doublement progressif des filtres)
- Global Average Pooling
- Intégration du SNR en fin de réseau
- Meilleure pour capturer des patterns complexes, mais plus coûteux

#### CNN progressif simple
- Séquence de Conv1D avec réduction progressive de la dimension temporelle
- Plus léger mais moins performant

#### STFT-CNN-LSTM
- Transformation en domaine fréquentiel via STFT
- CNN 2D sur le spectrogramme
- **Performances médiocres** : la représentation fréquentielle n'a pas apporté d'amélioration

## 4. Entraînement

### 4.1 Hyperparamètres
- **Batch size** : 512
- **Optimizer** : Adam
- **Loss** : CrossEntropyLoss
- **Epochs** : 500 (avec checkpoints tous les 50 epochs)
- **Device** : GPU (CUDA)

### 4.2 Gestion du SNR

Trois approches testées :

1. **Inclusion du SNR dans le modèle** (approche retenue)
   - Le modèle reçoit la valeur du SNR en input
   - Permet au modèle d'adapter sa prédiction selon le niveau de bruit

2. **Exclusion des échantillons SNR=0 dB**
   - Testée mais abandonnée
   - Dégrade légèrement les performances globales

3. **Entraînement sur un seul SNR**
   - Forte dégradation de la généralisation
   - Non recommandé

### 4.3 Suivi des entrainements et validation
- TensorBoard pour le monitoring en temps réel
- Métriques par SNR et par classe à chaque epoch
- Sauvegarde des checkpoints réguliers

## 5. Résultats

### 5.1 Performance globale

#### Configuration : SNR en input + Augmentation
```
Test Accuracy : 84.65%
Test Loss     : 0.3068
```

**Accuracy par classe** :
- Classe 0 : 87.51%
- Classe 1 : 75.33%
- Classe 2 : 88.99%
- Classe 3 : 87.72%
- Classe 4 : 78.68%
- Classe 5 : 89.66%

**Accuracy par SNR** :
- **SNR 0 dB**  : 40.62% 
- **SNR 10 dB** : 96.73%
- **SNR 20 dB** : 99.44%
- **SNR 30 dB** : 100.00%

#### Configuration : SNR en input sans augmentation
```
Test Accuracy : 86.57%
Test Loss     : 0.2786
```

**Accuracy par SNR** :
- **SNR 0 dB**  : 46.53%
- **SNR 10 dB** : 98.15%
- **SNR 20 dB** : 99.96%
- **SNR 30 dB** : 100.00%

### 5.2 Analyse des résultats

#### Points forts
1. **Excellentes performances à SNR élevé** : >99% d'accuracy à 20 et 30 dB
2. **Robustesse au bruit** : Même à 0 dB, le modèle atteint ~45% d'accuracy (vs 16.7% aléatoire)
3. **Équilibre entre classes** : Pas de déséquilibre majeur (75-90%)

#### Points d'attention
1. **SNR 0 dB** : Performance limitée mais attendue (signal fortement dégradé)
2. **Matrice de confusion** : Confusions principalement entre classes proches
   - Classes 0-1-2 (modulations PSK)
   - Classes 3-4-5 (modulations QAM/GMSK)

## 6. Evaluation sur le test set

Le script `pred_test.py` offre une évaluation complète :
- Prédictions sur le test set
- Métriques globales et par SNR/classe
- Génération automatique de graphiques
- Matrices de confusion détaillées
- Sauvegarde des résultats (.npz)

## 7. Conclusions et Perspectives


Le projet a permis de développer un système de classification robuste atteignant 84-86% d'accuracy globale, avec d'excellentes performances à SNR élevé (>99% à 20-30 dB).

Apports de l'approche:  
- L'intégration du SNR comme feature améliore significativement les performances
- L'augmentation de données accélère l'entraînement et améliore la généralisation
- L'architecture CNN-LSTM capture efficacement les patterns temporels


## 8. Références du Code

### Structure des fichiers
```
.
├── train.py              # Script d'entraînement principal
├── pred_test.py          # Évaluation sur test set
├── dataset.py            # DataLoader et augmentation
├── models.py             # Architecture CNN-LSTM-SNR
├── Other_model.py        # ResNet et modèles STFT
├── dump_model.py         # Modèle CNN simple
├── utils.py              # Utilitaires
├── README.md             # Résultats et remarques
├── test_results          # Résultats des modèles
└── .gitignore            # Configuration Git
```

### Commandes principales

**Entraînement** :
```bash
python train.py --name training
python train.py --load runs/checkpoints/checkpoint.pt  # Reprendre depuis checkpoint
```

**Évaluation** :
```bash
python pred_test.py --checkpoint runs/20241121-123456_run/checkpoint.pt \
                    --test test.hdf5 \
                    --class_names BPSK QPSK 8PSK 16QAM 64QAM GMSK
```

---

