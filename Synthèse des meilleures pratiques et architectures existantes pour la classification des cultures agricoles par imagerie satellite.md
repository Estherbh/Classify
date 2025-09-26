
## Synthèse des meilleures pratiques et architectures existantes pour la classification des cultures agricoles par imagerie satellite

Les recherches ont confirmé que l'imagerie satellite Sentinel-2, l'indice de végétation par différence normalisée (NDVI), et les algorithmes d'apprentissage automatique tels que Random Forest (RF) et XGBoost sont des outils couramment utilisés et efficaces pour la classification des cultures agricoles. Plusieurs études mettent en évidence l'intérêt de combiner RF et XGBoost pour améliorer la précision de détection.

### Points clés relevés :

*   **Imagerie Sentinel-2 et NDVI** : Sentinel-2 est une source de données gratuite et accessible, fournissant des images multi-spectrales à haute résolution spatiale et temporelle, idéales pour le suivi des cultures. Le NDVI, calculé à partir des bandes rouges et proche infrarouges de Sentinel-2, est un indicateur essentiel de la santé et de la densité de la végétation, largement utilisé pour la classification des cultures.

*   **Random Forest (RF) et XGBoost** : Ces deux algorithmes basés sur les arbres de décision sont très performants pour les tâches de classification. RF est robuste au surapprentissage et gère bien les données non linéaires, tandis que XGBoost est un algorithme de boosting qui excelle en termes de précision et de vitesse. Leur combinaison ou leur utilisation en ensemble peut potentiellement améliorer les résultats de classification.

*   **Approches à faible coût / budget zéro** :
    *   **Données** : Utilisation exclusive des données Sentinel-2 (via Google Earth Engine ou API Copernicus) qui sont gratuites.
    *   **Environnement de calcul** : Google Colab est une plateforme gratuite qui offre un accès à des GPU/TPU, ce qui est crucial pour le traitement de grandes quantités de données satellitaires et l'entraînement de modèles ML sans coût d'infrastructure.
    *   **Bibliothèques** : Utilisation de bibliothèques Python open-source telles que `scikit-learn` (pour Random Forest), `xgboost`, `pandas`, `numpy`, `matplotlib`, `geopandas`, `rasterio`, `folium` (pour la visualisation cartographique interactive) et `fpdf` (pour l'export PDF).

### Architecture optimisée (concept initial) :

L'architecture générale pour un système à budget zéro sur Google Colab pourrait ressembler à ceci :

1.  **Acquisition et Prétraitement des Données** :
    *   Accès aux données Sentinel-2 via Google Earth Engine (GEE) API (nécessite une authentification GEE, mais l'utilisation est gratuite pour la recherche et le développement). GEE permet de filtrer, de mosaïquer et de calculer le NDVI directement sur le cloud.
    *   Téléchargement des données prétraitées (NDVI et autres bandes pertinentes) pour les zones d'intérêt et les périodes avant/après 2020.
    *   Alignement des données satellitaires avec les polygones fournis par les agents de terrain.

2.  **Extraction des Caractéristiques** :
    *   Calcul du NDVI et potentiellement d'autres indices de végétation (EVI, SAVI) ou caractéristiques spectrales à partir des bandes Sentinel-2.
    *   Agrégation des valeurs (moyenne, médiane, écart-type) de ces indices sur les polygones d'intérêt pour chaque période.
    *   Création de séries temporelles de NDVI pour capturer l'évolution phénologique des cultures.

3.  **Modélisation Machine Learning** :
    *   **Préparation des données** : Division des données en ensembles d'entraînement et de test.
    *   **Entraînement des modèles** : Entraînement de Random Forest et XGBoost séparément ou en tant qu'ensemble (stacking/blending).
    *   **Calibration des probabilités** : Utilisation de techniques comme la régression isotonique ou la mise à l'échelle de Platt pour calibrer les probabilités de sortie des modèles, assurant que les probabilités reflètent la vraie confiance.

4.  **Système de Confiance** :
    *   Définition de seuils de probabilité pour classer les prédictions en 5 niveaux de confiance (par exemple, très faible, faible, moyenne, élevée, très élevée).
    *   Association d'actions recommandées à chaque niveau de confiance (par exemple, 



### Meilleures pratiques américaines et outils gratuits :

Les pratiques américaines en matière de classification des cultures par satellite, notamment celles du USDA (Département de l'Agriculture des États-Unis), mettent l'accent sur l'utilisation de l'imagerie satellite (Landsat et Sentinel) pour la cartographie des cultures. Leurs approches sont souvent basées sur des données à haute résolution temporelle et spatiale, et intègrent des techniques de machine learning.

Pour un budget zéro, les outils et plateformes suivants sont essentiels et compatibles avec Google Colab :

*   **Google Earth Engine (GEE)** : C'est une plateforme cloud de traitement de données géospatiales qui donne accès à de vastes catalogues de données satellitaires (y compris Sentinel-2) et offre des capacités de calcul puissantes. L'API Python de GEE est parfaitement intégrable dans Google Colab, permettant de réaliser l'acquisition, le filtrage, le calcul du NDVI et l'extraction de caractéristiques directement sur le cloud sans coût pour l'utilisateur.
*   **Bibliothèques Python open-source** :
    *   `ee` (Google Earth Engine Python API) : Pour l'interaction avec GEE.
    *   `numpy` et `pandas` : Pour la manipulation et l'analyse des données tabulaires.
    *   `scikit-learn` : Pour l'implémentation de Random Forest et d'autres algorithmes de ML, ainsi que pour la calibration des probabilités.
    *   `xgboost` : Pour l'implémentation de l'algorithme XGBoost.
    *   `matplotlib` et `seaborn` : Pour la visualisation des données.
    *   `geopandas` : Pour la manipulation des données géospatiales (polygones).
    *   `rasterio` : Pour la lecture et l'écriture de données raster.
    *   `folium` ou `ipyleaflet` : Pour la création de cartes interactives dans Google Colab.
    *   `fpdf` ou `reportlab` : Pour la génération de rapports PDF.
    *   `geojson` : Pour l'exportation au format GeoJSON.

L'intégration de GEE avec Google Colab est une approche très efficace pour gérer les données satellitaires à grande échelle sans nécessiter de ressources de calcul locales importantes, ce qui est parfait pour un projet à budget zéro. Des tutoriels et des exemples existent pour l'extraction de séries temporelles de NDVI et la classification des cultures en utilisant cette combinaison.



### Analyse du fichier `bd_ml.xlsx`

Le fichier `bd_ml.xlsx` contient les colonnes suivantes, qui seront utilisées pour la conception de l'architecture du modèle :

*   `gid_plantation`: Identifiant unique de la plantation.
*   `area`: Surface de la plantation.
*   `year_creation`: Année de création de la plantation. Cette colonne est essentielle pour la distinction avant/après 2020.
*   `culture`: Type de culture (palmier à huile, cacao, forêt). C'est la variable cible pour la classification.
*   `gid_town`: Identifiant de la ville.
*   `geom_json`: Représentation GeoJSON des polygones des plantations. C'est la donnée géographique clé pour l'extraction des images Sentinel-2.
*   `coordx`, `coordy`: Coordonnées géographiques (probablement le centroïde du polygone).
*   `name_country`: Pays de la plantation.
*   `id_town`, `name_town`, `id_contact`, `plantation_town`, `surface_ha`: Informations supplémentaires sur la localisation et la taille.

### Conception de l'architecture optimisée (détaillée)

En se basant sur les recherches et l'analyse des données, voici une architecture optimisée pour le modèle de machine learning, en gardant à l'esprit le budget zéro et l'exécution sur Google Colab :

**Phase 1: Acquisition et Prétraitement des Données Satellitaires (Google Earth Engine)**

*   **Accès aux données Sentinel-2 via GEE Python API** :
    *   Utiliser l'API `ee` dans Google Colab pour se connecter à Google Earth Engine.
    *   Pour chaque polygone (`geom_json`) et pour les périodes avant (par exemple, 2018-2019) et après (par exemple, 2021-2022) 2020, collecter les images Sentinel-2.
    *   Filtrer les images par date et par couverture nuageuse (par exemple, < 10%).
    *   Calculer le NDVI pour chaque image : `NDVI = (NIR - Red) / (NIR + Red)` où NIR est la bande 8 et Red est la bande 4 de Sentinel-2.
    *   Calculer d'autres indices si nécessaire (EVI, NDWI, etc.) pour enrichir les caractéristiques.
    *   Extraire des statistiques (moyenne, médiane, écart-type, min, max, quantiles) du NDVI et des autres bandes spectrales pour chaque polygone et pour chaque période (avant/après 2020). Cela transformera les données d'image en caractéristiques tabulaires.
    *   Gérer les données manquantes (par exemple, par interpolation temporelle si des séries temporelles complètes sont nécessaires, ou en excluant les polygones avec trop de données manquantes).

**Phase 2: Préparation des Données pour le Machine Learning**

*   **Fusion des données** :
    *   Combiner les caractéristiques extraites de GEE (statistiques NDVI, etc.) avec les informations du fichier `bd_ml.xlsx` (culture, année de création, etc.).
    *   Créer une colonne binaire `periode_apres_2020` (1 si `year_creation` >= 2020, 0 sinon) ou utiliser `year_creation` directement comme caractéristique numérique.
*   **Encodage des variables catégorielles** : Si d'autres variables catégorielles sont utilisées (par exemple, `name_country`), les encoder (One-Hot Encoding).
*   **Division des données** : Séparer les données en ensembles d'entraînement et de test (par exemple, 80% entraînement, 20% test).

**Phase 3: Modélisation Machine Learning (Random Forest + XGBoost)**

*   **Entraînement des modèles** :
    *   Entraîner un modèle Random Forest (`sklearn.ensemble.RandomForestClassifier`).
    *   Entraîner un modèle XGBoost (`xgboost.XGBClassifier`).
    *   **Hyperparamètres** : Utiliser une recherche de grille (`GridSearchCV`) ou une recherche aléatoire (`RandomizedSearchCV`) pour optimiser les hyperparamètres de chaque modèle. Cela peut être gourmand en calcul, donc commencer avec des grilles plus petites.
*   **Combinaison des modèles (Ensemble Learning)** :
    *   **Stacking/Blending** : Entraîner un méta-modèle (par exemple, une régression logistique simple) sur les prédictions (probabilités) des modèles Random Forest et XGBoost. Cela permet de combiner les forces des deux algorithmes.
*   **Calibration automatique des probabilités** :
    *   Appliquer des méthodes de calibration comme `CalibratedClassifierCV` de scikit-learn (avec `method='isotonic'` ou `method='sigmoid'`) sur les probabilités de sortie du modèle combiné. Cela garantit que les probabilités prédites reflètent fidèlement la probabilité réelle d'appartenance à une classe.

**Phase 4: Système de Confiance et Actions Recommandées**

*   **Définition des niveaux de confiance** : Baser les 5 niveaux de confiance sur les probabilités calibrées du modèle. Par exemple :
    *   Niveau 1 (Très faible confiance) : Probabilité < 0.2
    *   Niveau 2 (Faible confiance) : 0.2 <= Probabilité < 0.4
    *   Niveau 3 (Confiance moyenne) : 0.4 <= Probabilité < 0.6
    *   Niveau 4 (Confiance élevée) : 0.6 <= Probabilité < 0.8
    *   Niveau 5 (Très haute confiance) : Probabilité >= 0.8
*   **Actions recommandées** : Associer des actions spécifiques à chaque niveau de confiance. Par exemple :
    *   Niveau 1/2 : Nécessite une vérification sur le terrain ou une analyse visuelle approfondie par un expert.
    *   Niveau 3 : Vérification visuelle recommandée si possible.
    *   Niveau 4/5 : Confiance élevée, pas d'action immédiate requise, peut être utilisé pour la cartographie automatique.

**Phase 5: Dashboard Temps Réel et Exports**

*   **Dashboard (Google Colab / Streamlit)** :
    *   Utiliser `folium` ou `ipyleaflet` pour afficher les polygones sur une carte interactive.
    *   Colorer les polygones en fonction de la classe prédite et de la probabilité de confiance (par exemple, un dégradé de couleur).
    *   Afficher les informations clés (culture prédite, probabilité, niveau de confiance, actions recommandées) au survol ou au clic.
    *   Utiliser `ipywidgets` pour des contrôles interactifs (filtrage par culture, par niveau de confiance, par période).
    *   Pour un 


dashboard plus élaboré et potentiellement déployable, Streamlit est une option, mais cela pourrait dépasser le cadre strict de Google Colab sans déploiement externe.
*   **Design moderne dark-mode responsive** : Utiliser les options de style de `folium` ou `ipyleaflet` pour un thème sombre. Le responsive sera géré par la nature même de l'affichage dans Colab ou Streamlit.

*   **Exports CSV/PDF/GeoJSON** :
    *   **CSV** : Exporter les résultats (ID plantation, culture prédite, probabilité, niveau de confiance, actions recommandées) dans un fichier CSV en utilisant `pandas`.
    *   **PDF** : Générer un rapport PDF récapitulatif pour chaque polygone ou pour l'ensemble des résultats, incluant les informations clés et potentiellement une petite carte de la zone. Utiliser `fpdf` ou `reportlab`.
    *   **GeoJSON** : Exporter les polygones avec les attributs de classification (culture prédite, probabilité, confiance) au format GeoJSON en utilisant `geopandas`.

**Phase 6: Intégration et Tests**

*   **Intégration** : Assembler tous les composants dans un notebook Google Colab.
*   **Tests** : Tester le pipeline complet avec les données fournies et s'assurer de l'absence de bugs.

Cette architecture permet de répondre à toutes les exigences du projet avec un budget zéro, en tirant parti des outils open-source et des plateformes gratuites.

