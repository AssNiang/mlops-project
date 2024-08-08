# mlops-project
MLOps EPT


# Projet de Classification de Textes de Biens de Consommation

## Description

Ce projet est une application de classification de biens de consommation utilisant des techniques de traitement du langage naturel (NLP) et de machine learning. Il comprend des étapes de prétraitement des données, d'entraînement de modèles, et d'évaluation de performance. L'interface utilisateur est construite avec Streamlit pour la classification en temps réel.

## Support de présentation canva

[Présentation MLOps : Classifiez des biens de consommation](https://www.canva.com/design/DAGNOgBf44g/YZKTTFNj6TNruYDTkm3kTA/edit?utm_content=DAGNOgBf44g&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


## Structure du Repository

Voici un aperçu des principaux dossiers et fichiers dans ce repository :

```
.
├── .github/
│   └── workflows/
│       └── pipeline.yml              # Définition des workflows CI/CD pour GitHub Actions
├── .gitignore                        # Liste des fichiers et dossiers à ignorer par Git
├── .pylintrc                         # Configuration de pylint pour l'analyse statique du code
├── README.md                         # Ce fichier
├── api/
│   └── main.py                       # Point d'entrée principal pour l'API (FastAPI)
├── data/
│   └── ecommerceDataset.csv          # Dataset utilisé pour l'entraînement et la validation
├── ecommerce_text_classification.sh  # Script shell pour l'exécution du notebook. Le résultat esst enregistré dans le dossier logs.
├── logs/
│   └── 2024/08/
│       └── 20240808-0120-e-commerce-text-classification-tf-idf-artifact.ipynb   # Notebook de log pour la classification de texte
├── models/
│   ├── classifier.pkl                # Modèle de classification entraîné
│   └── tfidf_vectorizer.pkl          # Vectoriseur TF-IDF utilisé pour la transformation des textes
├── notebooks/
│   └── e-commerce-text-classification-tf-idf.ipynb   # Notebook pour la classification de texte
├── pages/
│   ├── classify.py                   # Page Streamlit pour la classification des biens
│   └── dataset.py                    # Page Streamlit pour afficher le dataset
├── pyproject.toml                    # Configuration du projet Python (dépendances, etc.)
├── reports/                          # Divers rapports et graphiques
│   ├── class_frequencies_pie_chart.png
│   ├── comparison_train_val_test_sizes_pie_chart.png
│   ├── confusion_matrix.png
│   ├── distribution_avg_word_length.png
│   ├── distribution_characters.png
│   └── distribution_words.png
│       
├── requirements.txt                  # Liste des dépendances Python du projet
├── settings/
│   └── params.py                     # Paramètres de configuration pour le projet
├── src/
│   ├── explore.py                    # Scripts pour l'exploration des données
│   ├── make_dataset.py               # Scripts pour charger les données
│   ├── preprocess.py                 # Scripts pour la préparation des données et la normalisation des textes
│   ├── train.py                      # Scripts pour l'entraînement des modèles
│   └── utils.py                      # Fonctions utilitaires pour le projet
├── streamlit_deploy.py               # Point d'entrée pour l'application Streamlit
├── tests/
│   ├── test_dataset.py               # Tests pour les fonctionnalités liées aux données
│   └── test_text_normalizer.py       # Tests pour la normalisation des textes
```

## Installation

1. Clonez le repository :

   ```bash
   git clone https://github.com/AssNiang/mlops-project.git
   cd mlops-project
   ```

2. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### Lancer l'application Streamlit

1. Assurez-vous que les modèles et les fichiers de données sont présents dans les répertoires appropriés.
2. Lancez l'application Streamlit avec :

   ```bash
   streamlit run streamlit_deploy.py
   ```

### Lancer l'API

Pour démarrer l'API FastAPI, utilisez la commande suivante :

   ```bash
   fastapi dev api/main.py
   ```

### Lancer MLflow

Pour démarrer l'interface utilisateur de MLflow pour suivre les expériences, utilisez la commande suivante :

   ```bash
   mlflow ui
   ```

### Utilisation des Notebooks

Après l'installation des dépendances, vous pouvez utiliser directement les notebooks présents dans le répertoire `notebooks/` pour explorer et analyser les données.

Alternativement, vous pouvez exécuter le script `ecommerce_text_classification.sh` pour lancer le processus de classification :

   ```bash
   bash ecommerce_text_classification.sh
   ```

Dans ce fichier shell, nous utilisons la bibliothèque `papermill` pour exécuter le notebook et sauvegarder l'artifact dans le dossier des logs (il sera automatiquement créé si nécessaire).


## Contribuer

Les contributions sont les bienvenues !

## Licence



## Contact

Pour toute question, veuillez me contacter par mail sur nianga@ept.sn.

## Références
https://www.kaggle.com/code/sugataghosh/e-commerce-text-classification-tf-idf-word2vec
