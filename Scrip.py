import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(file_path, sep, numeric_features, categorical_features):
    """
    Charge et prétraite les données.
    - Gère les valeurs manquantes.
    - Encode les colonnes catégorielles.
    - Standardise les colonnes numériques.

    Parameters:
    - file_path (str): Chemin du fichier CSV.
    - sep (str): Délimiteur des colonnes dans le fichier CSV.
    - numeric_features (list): Colonnes numériques.
    - categorical_features (list): Colonnes catégoriques.

    Returns:
    - data (DataFrame): Données prétraitées.
    - features (array): Matrice de caractéristiques transformée.
    """
    data = pd.read_csv(file_path, sep=sep)
    data['Income'] = data['Income'].fillna(data['Income'].mean())
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
    data['Customer_Seniority'] = (pd.to_datetime('today') - data['Dt_Customer']).dt.days
    data['Age'] = pd.to_datetime('today').year - data['Year_Birth']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    features = preprocessor.fit_transform(data)
    return data, features


def elbow_method(features, max_clusters=10):
    """
    Applique la méthode du coude pour déterminer le nombre optimal de clusters.

    Parameters:
    - features (array): Matrice de caractéristiques.
    - max_clusters (int): Nombre maximal de clusters.

    Returns:
    - inertia (list): Liste des inerties pour chaque nombre de clusters.
    """
    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_clusters + 1), inertia, 'bx-')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude')
    plt.show()
    return inertia


def apply_kmeans(features, n_clusters):
    """
    Applique K-Means sur les données.

    Parameters:
    - features (array): Matrice de caractéristiques.
    - n_clusters (int): Nombre de clusters.

    Returns:
    - kmeans (KMeans): Modèle K-Means ajusté.
    - clusters (array): Clusters prédits.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return kmeans, clusters


def apply_hierarchical_clustering(features, n_clusters):
    """
    Applique le clustering hiérarchique.

    Parameters:
    - features (array): Matrice de caractéristiques.
    - n_clusters (int): Nombre de clusters.

    Returns:
    - agg_model (AgglomerativeClustering): Modèle hiérarchique ajusté.
    - clusters (array): Clusters prédits.
    """
    agg_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = agg_model.fit_predict(features)
    return agg_model, clusters


def plot_dendrogram(features):
    """
    Trace un dendrogramme pour le clustering hiérarchique.

    Parameters:
    - features (array): Matrice de caractéristiques.
    """
    Z = linkage(features, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title('Dendrogramme du clustering hiérarchique')
    plt.xlabel('Points de données')
    plt.ylabel('Distance')
    plt.show()


def visualize_clusters(features, clusters, title):
    """
    Réduit la dimensionnalité à 2 composantes principales et visualise les clusters.

    Parameters:
    - features (array): Matrice de caractéristiques.
    - clusters (array): Clusters prédits.
    - title (str): Titre du graphique.
    """
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    principal_df['Cluster'] = clusters
    sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
    plt.title(title)
    plt.show()


def evaluate_clustering(features, clusters):
    """
    Évalue la qualité du clustering avec des métriques.

    Parameters:
    - features (array): Matrice de caractéristiques.
    - clusters (array): Clusters prédits.

    Returns:
    - silhouette (float): Indice de silhouette.
    - davies_bouldin (float): Coefficient de Davies-Bouldin.
    """
    silhouette = silhouette_score(features, clusters)
    davies_bouldin = davies_bouldin_score(features, clusters)
    return silhouette, davies_bouldin


# Chemin des données et colonnes à utiliser
file_path = "C:/Users/HP PROBOOK/Desktop/Projet clustering/marketing_campaign_clean.csv"
categorical_features = ['Education', 'Marital_Status']
numeric_features = ['Age', 'Customer_Seniority', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 
                    'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

# Chargement et prétraitement des données
data, features = load_and_preprocess_data(file_path, sep=";", numeric_features=numeric_features, categorical_features=categorical_features)

# Méthode du coude pour K-Means
elbow_method(features)

# Application de K-Means avec 4 clusters
kmeans_model, kmeans_clusters = apply_kmeans(features, n_clusters=4)
visualize_clusters(features, kmeans_clusters, title="Clusters K-Means (PCA)")

# Clustering hiérarchique avec 3 clusters
plot_dendrogram(features)
hierarchical_model, hierarchical_clusters = apply_hierarchical_clustering(features, n_clusters=3)
visualize_clusters(features, hierarchical_clusters, title="Clusters Hiérarchiques (PCA)")

# Évaluation des clusters
silhouette_kmeans, davies_bouldin_kmeans = evaluate_clustering(features, kmeans_clusters)
silhouette_hierarchical, davies_bouldin_hierarchical = evaluate_clustering(features, hierarchical_clusters)

print(f"K-Means - Silhouette: {silhouette_kmeans}, Davies-Bouldin: {davies_bouldin_kmeans}")
print(f"Clustering Hiérarchique - Silhouette: {silhouette_hierarchical}, Davies-Bouldin: {davies_bouldin_hierarchical}")
