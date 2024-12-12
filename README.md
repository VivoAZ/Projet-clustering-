# Prédiction du prix 

## Description 
C'est un  projet de classification. Nous disposons d'une base de données issue d'une campagne marketing d'une entreprise qui vend plusieurs articles. L'objectif est de segmenter la clientèle en tenant compte du comportement des individus. 

## Fonctionnalité principale 
Regrouper les clients par catégorie afin de proposer des offres adaptées à chaque classe et optimiser le budget des campagnes.  

## Installation 

### 1- Cloner le dépôt : 
git clone https://github.com/VivoAZ/Projet-clustering-/tree/master.git 

cd Projet-clustering-/tree/master  

### 2- Créer et activer un environnement virtuel (venv) : 
python -m venv env 

source env/bin/activate  # Pour Linux/macOS 

env\Scripts\activate     # Pour Windows 

### 3- Installer les dépendances : 
pip install -r requirements.txt

## Exécution 
Commande pour lancer le projet 
python main.py 

N'oubliez pas de vérifier le chemin d'accès des fichiers main.py et HousingData.csv selon où vous les avez sauvegardés sur votre machine. 

## Structure du projet
main.py : Script principal pour l’entraînement et la prédiction du modèle. 

marketing_campaign_clean.csv : Contient les jeux de données bruts et transformés. 

gradient_boosting_model.pkl : Modèle sauvegardé au format pkl.

Clustering.ipynb : Notebook Jupyter pour l’analyse exploratoire et les tests. 

requirements.txt : Liste des dépendances nécessaires. 

## Données
Les informations proviennent de la plateforme publique Kaggle.

## Collaboration
Si vous souhaitez contribuer :

1- Forkez le projet. 

2- Créez une branche (git checkout -b ma-branche).

3- Soumettez une Pull Request. 
