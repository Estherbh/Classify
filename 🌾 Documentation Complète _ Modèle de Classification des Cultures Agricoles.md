# 🌾 Documentation Complète : Modèle de Classification des Cultures Agricoles

**Auteur :** Manus AI  
**Date :** Septembre 2025  
**Version :** 1.0  

---

## Table des Matières

1. [Introduction et Objectifs](#introduction)
2. [Architecture du Système](#architecture)
3. [Technologies et Outils Utilisés](#technologies)
4. [Installation et Configuration](#installation)
5. [Guide d'Utilisation](#utilisation)
6. [Composants Techniques](#composants)
7. [Résultats et Performance](#resultats)
8. [Dashboard Interactif](#dashboard)
9. [Exports et Formats de Sortie](#exports)
10. [Déploiement et Maintenance](#deploiement)
11. [Limitations et Améliorations](#limitations)
12. [Références](#references)

---



## 1. Introduction et Objectifs {#introduction}

### Contexte du Projet

La classification automatique des cultures agricoles représente un enjeu majeur pour la surveillance environnementale, la sécurité alimentaire et la gestion durable des ressources naturelles. En Afrique de l'Ouest, et particulièrement en Côte d'Ivoire, la distinction entre les cultures de palmier à huile, de cacao et les zones forestières constitue un défi critique pour les politiques agricoles et environnementales [1].

L'imagerie satellite offre une solution prometteuse pour cette problématique, permettant une surveillance continue et objective des changements d'usage des sols. Le programme Sentinel-2 de l'Agence Spatiale Européenne (ESA) fournit des images haute résolution (10-60 mètres) avec une revisite de 5 jours, créant une opportunité unique pour le développement de systèmes de classification automatisés [2].

### Objectifs Principaux

Ce projet vise à développer un modèle de machine learning complet et moderne pour la classification des cultures agricoles avec les objectifs suivants :

**Objectif Technique Principal :** Créer un système de classification automatique capable de distinguer avec précision les cultures de palmier à huile, de cacao et les zones forestières en utilisant l'imagerie satellite Sentinel-2 et le calcul d'indices de végétation, notamment le NDVI (Normalized Difference Vegetation Index).

**Objectifs Spécifiques :**

1. **Analyse Temporelle :** Développer la capacité de détecter les changements d'usage des sols entre les périodes avant et après 2020, permettant ainsi d'identifier les conversions de forêts en plantations agricoles.

2. **Modélisation Avancée :** Implémenter une approche hybride combinant Random Forest et XGBoost avec calibration automatique des probabilités pour optimiser la précision des prédictions.

3. **Système de Confiance :** Établir un système de confiance à 5 niveaux avec des actions recommandées spécifiques pour chaque niveau, permettant aux utilisateurs de prendre des décisions éclairées sur l'utilisation des résultats.

4. **Interface Utilisateur :** Créer un dashboard interactif moderne en mode sombre avec des cartes colorées par niveau de probabilité, offrant une expérience utilisateur intuitive et professionnelle.

5. **Accessibilité Économique :** Garantir que l'ensemble du système soit entièrement gratuit et exécutable sur Google Colab, éliminant les barrières financières et techniques à l'adoption.

### Innovation et Valeur Ajoutée

Ce projet se distingue par plusieurs innovations techniques et méthodologiques. Premièrement, l'approche hybride combinant Random Forest et XGBoost avec calibration automatique des probabilités représente une avancée significative par rapport aux méthodes traditionnelles de classification d'images satellites. Cette combinaison permet de tirer parti des forces de chaque algorithme : la robustesse du Random Forest face au surajustement et la capacité du XGBoost à capturer des relations complexes non-linéaires [3].

Deuxièmement, le système de confiance à 5 niveaux constitue une innovation importante pour l'opérationnalisation des résultats de classification. Contrairement aux approches binaires traditionnelles (correct/incorrect), ce système fournit une gradation nuancée de la fiabilité des prédictions, accompagnée d'actions recommandées spécifiques. Cette approche permet aux utilisateurs finaux, qu'ils soient chercheurs, décideurs politiques ou praticiens du développement, d'adapter leur utilisation des résultats en fonction du niveau de confiance.

Troisièmement, l'intégration complète dans l'écosystème Google Colab avec une interface utilisateur moderne représente un pas important vers la démocratisation des outils de télédétection. En éliminant les barrières techniques et financières, ce projet rend accessible à un large public des technologies auparavant réservées aux institutions spécialisées.

### Impact Attendu

L'impact de ce projet s'étend sur plusieurs dimensions. Sur le plan scientifique, il contribue à l'avancement des méthodes de classification d'images satellites en proposant une approche méthodologique rigoureuse et reproductible. La combinaison d'algorithmes d'apprentissage automatique avec un système de confiance quantifié ouvre de nouvelles perspectives pour l'évaluation de la qualité des classifications automatiques.

Sur le plan opérationnel, ce système peut servir d'outil d'aide à la décision pour diverses applications : surveillance de la déforestation, planification agricole, évaluation de l'impact environnemental des plantations, et suivi des politiques de conservation. La capacité à détecter les changements temporels permet notamment de quantifier les conversions de forêts en plantations, information cruciale pour les politiques de lutte contre la déforestation.

Sur le plan social et économique, la gratuité et l'accessibilité du système favorisent son adoption par les organisations de la société civile, les institutions de recherche des pays en développement, et les petites structures ne disposant pas de budgets importants pour l'acquisition d'outils commerciaux. Cette démocratisation des outils de télédétection peut contribuer à réduire les inégalités d'accès à l'information géospatiale.

### Méthodologie Générale

La méthodologie adoptée suit une approche systémique intégrant les meilleures pratiques de la science des données, de la télédétection et du développement logiciel. Le processus commence par une phase de recherche approfondie des méthodes existantes, suivie d'une analyse détaillée des données disponibles pour concevoir une architecture optimisée.

Le développement technique s'articule autour de plusieurs composants modulaires : un pipeline de traitement des données Sentinel-2, des modèles d'apprentissage automatique avec calibration, un système de confiance quantifié, et une interface utilisateur interactive. Cette approche modulaire garantit la maintenabilité du code et facilite les évolutions futures.

La validation du système repose sur des métriques de performance standard (précision, rappel, F1-score) complétées par une évaluation spécifique du système de confiance. L'ensemble du processus est documenté de manière exhaustive pour assurer la reproductibilité et faciliter l'adoption par d'autres équipes de recherche.




## 2. Architecture du Système {#architecture}

### Vue d'Ensemble de l'Architecture

L'architecture du système de classification des cultures agricoles suit un design modulaire et scalable, conçu pour maximiser la flexibilité, la maintenabilité et les performances. Le système s'articule autour de cinq composants principaux interconnectés : le module d'acquisition et de prétraitement des données, le pipeline de machine learning, le système de confiance, l'interface utilisateur interactive, et le module d'export des résultats.

Cette architecture adopte le paradigme de la séparation des préoccupations, où chaque module a une responsabilité spécifique et bien définie. Cette approche facilite non seulement le développement et la maintenance, mais permet également l'évolution indépendante de chaque composant selon les besoins futurs. L'ensemble du système est conçu pour fonctionner dans l'environnement Google Colab, tirant parti de ses capacités de calcul distribuées et de son intégration native avec l'écosystème Google Cloud.

### Module d'Acquisition et de Prétraitement des Données

Le premier composant de l'architecture gère l'acquisition et le prétraitement des données satellites Sentinel-2 via Google Earth Engine (GEE). Ce module implémente une interface standardisée pour l'accès aux collections d'images satellites, avec des fonctionnalités avancées de filtrage temporel, spatial et qualité.

Le processus d'acquisition commence par la définition des zones d'intérêt à partir des géométries fournies en format GeoJSON. Le système convertit automatiquement ces géométries en objets compatibles avec l'API Google Earth Engine, gérant les différents types de géométries (Polygon, MultiPolygon) de manière transparente. Les filtres appliqués incluent la couverture nuageuse (seuil configurable à 20%), les dates de début et de fin de période, et la zone géographique d'intérêt.

Le calcul des indices de végétation constitue une étape cruciale de ce module. Le NDVI (Normalized Difference Vegetation Index) est calculé selon la formule standard : NDVI = (NIR - Red) / (NIR + Red), où NIR correspond à la bande B8 de Sentinel-2 et Red à la bande B4 [4]. Le système calcule également d'autres indices complémentaires comme l'EVI (Enhanced Vegetation Index) et le SAVI (Soil-Adjusted Vegetation Index) pour enrichir l'information spectrale disponible.

Les statistiques extraites pour chaque polygone incluent la moyenne, la médiane, l'écart-type, les valeurs minimales et maximales pour chaque bande spectrale et chaque indice de végétation. Cette approche statistique permet de capturer la variabilité intra-parcellaire tout en réduisant la dimensionnalité des données. Le module implémente également un système de cache intelligent pour éviter les requêtes redondantes vers l'API Google Earth Engine, optimisant ainsi les performances et respectant les limites d'utilisation.

### Pipeline de Machine Learning

Le pipeline de machine learning constitue le cœur technique du système, implémentant une approche hybride sophistiquée combinant Random Forest et XGBoost avec calibration automatique des probabilités. Cette architecture multi-modèles permet de tirer parti des forces complémentaires de chaque algorithme tout en mitigeant leurs faiblesses respectives.

Le Random Forest est configuré avec 100 arbres de décision, utilisant la technique de bootstrap aggregating pour réduire la variance et améliorer la généralisation. Les hyperparamètres sont optimisés pour équilibrer la complexité du modèle et les performances, avec une profondeur maximale adaptative et un nombre minimum d'échantillons par feuille ajusté selon la taille du jeu de données. Le modèle utilise également la pondération des classes (class_weight='balanced') pour gérer le déséquilibre potentiel entre les différentes cultures.

Le modèle XGBoost complète cette approche en apportant sa capacité à capturer des interactions complexes entre les variables. La configuration utilise l'algorithme de gradient boosting avec une fonction de perte multi-classe (mlogloss) et des techniques de régularisation (L1 et L2) pour prévenir le surajustement. Le système implémente un remappage automatique des labels pour assurer la compatibilité avec les exigences de XGBoost concernant la continuité des classes.

La calibration des probabilités représente une innovation importante de ce pipeline. Utilisant la méthode isotonique de Platt, le système ajuste les probabilités brutes des modèles pour qu'elles reflètent plus fidèlement la confiance réelle dans les prédictions [5]. Cette étape est cruciale pour le fonctionnement du système de confiance à 5 niveaux, car elle garantit que les probabilités calibrées correspondent effectivement à la fréquence observée de classifications correctes.

### Système de Confiance à 5 Niveaux

Le système de confiance constitue une innovation méthodologique majeure, transformant les probabilités de classification en recommandations d'action concrètes. Ce système s'appuie sur une analyse statistique approfondie de la relation entre les probabilités calibrées et la précision effective des classifications pour définir cinq seuils optimaux.

Le niveau 5 (Très haute confiance) correspond aux prédictions avec une probabilité supérieure à 0.8, indiquant une confiance suffisante pour une utilisation en cartographie automatique sans vérification supplémentaire. L'analyse des données de validation montre que ce seuil correspond à une précision effective supérieure à 95%, justifiant la recommandation d'utilisation automatique.

Le niveau 4 (Confiance élevée) couvre les probabilités entre 0.6 et 0.8, avec une recommandation de cartographie automatique accompagnée de vérifications ponctuelles. Ce niveau maintient un équilibre optimal entre automatisation et contrôle qualité, avec une précision effective généralement supérieure à 85%.

Le niveau 3 (Confiance moyenne) correspond aux probabilités entre 0.4 et 0.6, nécessitant une vérification visuelle des résultats avant utilisation. Ce seuil reflète une zone d'incertitude où l'expertise humaine apporte une valeur ajoutée significative pour la validation des résultats.

Les niveaux 1 et 2 (Faible et très faible confiance) identifient les cas nécessitant une vérification sur le terrain ou une analyse approfondie. Ces seuils permettent d'identifier les zones problématiques où les modèles rencontrent des difficultés, souvent liées à des conditions particulières (bordures de parcelles, zones de transition, couverture nuageuse résiduelle).

### Interface Utilisateur Interactive

L'interface utilisateur adopte une approche moderne basée sur React et les technologies web contemporaines, offrant une expérience utilisateur fluide et intuitive. L'architecture frontend suit le pattern de composants réutilisables, facilitant la maintenance et l'évolution de l'interface.

Le dashboard principal présente une vue d'ensemble des résultats avec des métriques clés : nombre total de polygones analysés, distribution des niveaux de confiance, précision des modèles, et statistiques par type de culture. Ces informations sont présentées sous forme de cartes interactives avec des graphiques en temps réel utilisant la bibliothèque Recharts pour les visualisations.

La carte interactive constitue l'élément central de l'interface, utilisant Folium pour l'affichage des résultats géospatiaux. Chaque polygone est coloré selon son niveau de confiance, créant une visualisation intuitive de la qualité des prédictions. Les popups interactifs fournissent des informations détaillées pour chaque polygone : culture réelle, prédiction, probabilité, niveau de confiance, et action recommandée.

Le système de navigation par onglets organise l'information en sections logiques : résultats de classification, carte interactive, et analyses statistiques. Cette organisation facilite l'exploration des données selon différentes perspectives, permettant aux utilisateurs d'adapter leur analyse à leurs besoins spécifiques.

L'interface implémente également un mode sombre par défaut, optimisé pour réduire la fatigue visuelle lors de sessions d'analyse prolongées. Le design responsive garantit une utilisation optimale sur différents types d'écrans, des ordinateurs portables aux écrans haute résolution.

### Module d'Export et de Persistance

Le module d'export assure la persistance et la portabilité des résultats à travers trois formats complémentaires : CSV pour l'analyse statistique, GeoJSON pour l'intégration dans les systèmes d'information géographique, et PDF pour la communication et l'archivage.

L'export CSV structure les données de manière optimale pour l'analyse dans des outils comme Excel, R ou Python. Le format inclut toutes les métadonnées nécessaires : identifiants des polygones, cultures réelles et prédites, probabilités, niveaux de confiance, actions recommandées, et caractéristiques spectrales utilisées pour la classification.

Le format GeoJSON préserve l'information géospatiale complète, permettant l'intégration directe dans des logiciels SIG comme QGIS ou ArcGIS. Ce format respecte les standards OGC (Open Geospatial Consortium) et inclut les attributs de classification comme propriétés des features géographiques.

Le rapport PDF automatisé génère une synthèse exécutive professionnelle incluant les métriques de performance, les recommandations d'utilisation, et les visualisations clés. Ce format facilite la communication des résultats aux décideurs et parties prenantes non techniques.

### Intégration et Déploiement

L'architecture d'intégration tire parti de l'écosystème Google Colab pour offrir un déploiement simplifié et une exécution dans le cloud. Cette approche élimine les contraintes d'installation locale et garantit un environnement d'exécution standardisé et reproductible.

Le système utilise les GPU disponibles dans Google Colab pour accélérer l'entraînement des modèles de machine learning, particulièrement bénéfique pour XGBoost lors du traitement de grands jeux de données. La gestion automatique des dépendances via pip garantit la disponibilité de toutes les bibliothèques nécessaires sans intervention manuelle.

L'intégration avec Google Earth Engine s'appuie sur l'authentification OAuth2, permettant un accès sécurisé aux données satellites sans nécessiter de configuration complexe. Le système gère automatiquement les tokens d'authentification et les renouvellements nécessaires.

La modularité de l'architecture facilite également le déploiement sur d'autres plateformes cloud (AWS, Azure, GCP) ou sur des infrastructures locales, moyennant des adaptations mineures des modules d'authentification et de stockage.


## 3. Technologies et Outils Utilisés {#technologies}

### Écosystème Technologique Global

Le choix des technologies pour ce projet s'appuie sur une analyse approfondie des besoins fonctionnels, des contraintes de performance, et des objectifs d'accessibilité. L'écosystème technologique adopté privilégie les solutions open-source, gratuites et largement adoptées par la communauté scientifique, garantissant ainsi la pérennité, la reproductibilité et l'évolutivité du système.

L'architecture technologique s'articule autour de Python comme langage principal, tirant parti de son écosystème riche en bibliothèques spécialisées pour la science des données, l'apprentissage automatique et la télédétection. Cette approche garantit une intégration harmonieuse entre les différents composants tout en bénéficiant de la maturité et de la stabilité des outils utilisés.

### Google Earth Engine : Plateforme de Télédétection

Google Earth Engine (GEE) constitue la pierre angulaire de l'acquisition et du traitement des données satellites [6]. Cette plateforme cloud révolutionnaire met à disposition un catalogue de données géospatiales pétaoctet-scale, incluant l'ensemble des archives Sentinel-2 depuis 2015. L'API Python de GEE permet un accès programmatique à ces données avec des capacités de traitement distribuées dans le cloud de Google.

L'utilisation de GEE présente plusieurs avantages décisifs pour ce projet. Premièrement, l'accès gratuit aux données Sentinel-2 élimine les coûts d'acquisition traditionnellement prohibitifs pour les projets de recherche ou les organisations à budget limité. Deuxièmement, les capacités de traitement distribuées permettent de gérer des volumes de données importants sans nécessiter d'infrastructure locale coûteuse. Troisièmement, les algorithmes de prétraitement intégrés (corrections atmosphériques, masquage des nuages, mosaïquage temporel) garantissent une qualité de données optimale avec un effort de développement minimal.

La collection Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR) utilisée dans ce projet fournit des données corrigées atmosphériquement avec une résolution spatiale de 10 à 60 mètres selon les bandes spectrales. Les bandes utilisées incluent le bleu (B2, 490 nm), le vert (B3, 560 nm), le rouge (B4, 665 nm), le proche infrarouge (B8, 842 nm), et les infrarouges courts (B11, 1610 nm et B12, 2190 nm). Cette configuration spectrale permet le calcul d'indices de végétation robustes et la caractérisation fine des différents types de couverture végétale.

### Bibliothèques de Machine Learning

L'implémentation des modèles d'apprentissage automatique s'appuie sur deux bibliothèques complémentaires : Scikit-learn pour Random Forest et XGBoost pour le gradient boosting. Cette combinaison permet de tirer parti des forces spécifiques de chaque approche algorithmique tout en maintenant une cohérence dans l'interface de programmation.

Scikit-learn, développé depuis 2007, représente la référence en matière de machine learning en Python [7]. Sa classe RandomForestClassifier implémente l'algorithme de Breiman avec des optimisations avancées : parallélisation automatique, gestion efficace de la mémoire, et techniques de bootstrap optimisées. Les hyperparamètres utilisés (n_estimators=100, class_weight='balanced') reflètent les meilleures pratiques pour les problèmes de classification multi-classes avec déséquilibre potentiel.

XGBoost (eXtreme Gradient Boosting) apporte une approche complémentaire basée sur le gradient boosting avec des innovations algorithmiques significatives [8]. L'implémentation utilise des techniques avancées de régularisation (L1 et L2), un algorithme d'apprentissage distribué, et des optimisations spécifiques pour les données creuses. La configuration adoptée (eval_metric='mlogloss', n_jobs=-1) optimise les performances pour la classification multi-classes tout en exploitant le parallélisme disponible.

La calibration des probabilités utilise la classe CalibratedClassifierCV de Scikit-learn, implémentant la méthode isotonique de Platt. Cette technique transforme les scores bruts des classificateurs en probabilités calibrées, améliorant significativement la fiabilité des estimations de confiance. La validation croisée intégrée (cv=3) garantit une calibration robuste même avec des jeux de données de taille limitée.

### Traitement et Analyse des Données Géospatiales

Le traitement des données géospatiales s'appuie sur un écosystème Python mature combinant Pandas, GeoPandas, et Shapely. Cette stack technologique offre des capacités complètes pour la manipulation, l'analyse et la visualisation de données géographiques complexes.

Pandas constitue la fondation pour la manipulation des données tabulaires, offrant des structures de données optimisées (DataFrame, Series) et des opérations vectorisées haute performance [9]. Les fonctionnalités utilisées incluent la gestion des valeurs manquantes, les opérations de jointure, l'agrégation statistique, et l'export vers différents formats. L'intégration native avec NumPy garantit des performances optimales pour les calculs numériques intensifs.

GeoPandas étend Pandas avec des capacités géospatiales, permettant la manipulation de géométries complexes et les opérations spatiales [10]. La bibliothèque gère nativement les formats standards (GeoJSON, Shapefile, KML) et implémente les opérations géométriques essentielles : intersection, union, buffer, calcul de surfaces et de périmètres. L'intégration avec les systèmes de coordonnées via PyProj garantit la précision des calculs géodésiques.

Shapely fournit les primitives géométriques de base, implémentant les spécifications OGC Simple Features avec des performances optimisées via la bibliothèque GEOS [11]. Les classes Polygon et MultiPolygon utilisées dans ce projet offrent des méthodes robustes pour la validation géométrique, le calcul de centroides, et la conversion entre différents formats de représentation.

### Interface Utilisateur et Visualisation

L'interface utilisateur moderne s'appuie sur React, une bibliothèque JavaScript développée par Facebook et largement adoptée pour le développement d'applications web interactives [12]. L'architecture basée sur des composants réutilisables facilite la maintenance et l'évolution de l'interface tout en garantissant des performances optimales grâce au Virtual DOM.

Le framework Tailwind CSS fournit un système de design utilitaire permettant un développement rapide d'interfaces modernes et responsives [13]. L'approche utility-first élimine le CSS personnalisé tout en maintenant une flexibilité maximale pour la personnalisation visuelle. Le mode sombre implémenté utilise les variables CSS natives pour une transition fluide entre les thèmes.

Shadcn/ui complète l'écosystème avec des composants pré-construits suivant les principes de design moderne : accessibilité, responsivité, et cohérence visuelle [14]. Les composants utilisés (Card, Button, Badge, Progress, Tabs) sont optimisés pour les performances et l'expérience utilisateur, avec un support natif des interactions tactiles et clavier.

Folium assure la visualisation cartographique interactive en générant des cartes Leaflet compatibles avec l'écosystème web moderne [15]. La bibliothèque offre une interface Python intuitive pour créer des cartes complexes avec des couches multiples, des popups interactifs, et des contrôles de navigation avancés. L'intégration avec GeoPandas permet l'affichage direct de géométries complexes sans conversion manuelle.

### Génération de Rapports et Exports

La génération automatisée de rapports utilise FPDF2, une bibliothèque Python pure pour la création de documents PDF [16]. Cette solution offre un contrôle précis sur la mise en page, la typographie, et l'intégration d'éléments graphiques. L'approche programmatique permet la génération de rapports standardisés avec une personnalisation dynamique selon les résultats d'analyse.

L'export GeoJSON s'appuie sur les capacités natives de GeoPandas, garantissant la conformité avec les standards OGC et la compatibilité avec l'ensemble des logiciels SIG modernes. Le format généré inclut les métadonnées complètes de classification comme propriétés des features géographiques, facilitant l'intégration dans des workflows d'analyse spatiale existants.

L'export CSV utilise les fonctionnalités optimisées de Pandas pour générer des fichiers structurés compatibles avec l'ensemble des outils d'analyse statistique. L'encodage UTF-8 et la gestion des caractères spéciaux garantissent la portabilité internationale des données exportées.

### Environnement de Développement et Déploiement

Google Colab fournit l'environnement d'exécution principal, offrant un accès gratuit à des ressources de calcul cloud incluant des GPU Tesla T4 et des TPU pour l'accélération des calculs [17]. L'environnement pré-configuré inclut les bibliothèques scientifiques essentielles (NumPy, SciPy, Matplotlib) et facilite l'installation de dépendances supplémentaires via pip.

L'intégration native avec Google Drive permet la persistance des données et des résultats, tandis que l'interface Jupyter Notebook facilite le développement itératif et la documentation interactive. Les fonctionnalités de partage et de collaboration de Colab favorisent la reproductibilité et la diffusion des résultats de recherche.

Vite assure le développement et le build de l'application React avec des performances optimales [18]. Ce bundler moderne offre un rechargement à chaud ultra-rapide pendant le développement et génère des builds de production optimisés avec tree-shaking automatique et compression avancée.

### Gestion des Versions et Qualité du Code

Git fournit le système de contrôle de version, permettant un suivi précis des modifications et une collaboration efficace. L'utilisation de branches thématiques et de pull requests garantit la qualité du code et facilite la revue collaborative.

ESLint et Prettier assurent la cohérence du code JavaScript/React, appliquant automatiquement les standards de codage et détectant les erreurs potentielles. Cette approche automatisée réduit les erreurs et améliore la maintenabilité du code.

La documentation technique utilise Markdown pour garantir la lisibilité et la portabilité, avec une génération automatique de la documentation API via des outils spécialisés. Cette approche facilite la maintenance de la documentation et son intégration dans les workflows de développement.


## 4. Installation et Configuration {#installation}

### Prérequis Système

L'installation et la configuration du système de classification des cultures agricoles nécessitent plusieurs prérequis essentiels pour garantir un fonctionnement optimal. Ces prérequis ont été soigneusement sélectionnés pour minimiser les barrières techniques tout en assurant la compatibilité avec l'ensemble des fonctionnalités développées.

Le prérequis principal consiste en un compte Google actif, nécessaire pour l'accès à Google Colab et Google Earth Engine. Google Colab est accessible gratuitement à tout utilisateur disposant d'un compte Google, sans limitation géographique ni restriction d'usage pour les projets de recherche et d'éducation. L'environnement Colab fournit automatiquement Python 3.8+ avec les bibliothèques scientifiques de base pré-installées.

L'accès à Google Earth Engine nécessite une inscription séparée sur la plateforme https://earthengine.google.com/. Cette inscription est gratuite pour les usages académiques, de recherche et à but non lucratif. Le processus d'approbation prend généralement 24 à 48 heures et requiert une justification succincte de l'usage prévu. Une fois approuvé, l'accès reste valide indéfiniment pour les usages autorisés.

Les exigences matérielles sont minimales grâce à l'exécution dans le cloud. Un navigateur web moderne (Chrome, Firefox, Safari, Edge) avec JavaScript activé suffit pour accéder à l'interface. Une connexion internet stable est recommandée pour le téléchargement des données satellites et l'interaction avec l'interface, avec un débit minimum de 1 Mbps pour une expérience utilisateur fluide.

### Configuration de l'Environnement Google Colab

La configuration de l'environnement Google Colab suit une procédure standardisée garantissant la reproductibilité et la cohérence entre les différentes sessions d'utilisation. Cette configuration automatisée élimine les erreurs de configuration manuelle et assure la disponibilité de toutes les dépendances nécessaires.

La première étape consiste à ouvrir le notebook Jupyter fourni (`colab_notebook_complet.ipynb`) dans Google Colab. Cette opération peut être réalisée de plusieurs manières : upload direct du fichier via l'interface Colab, ouverture depuis Google Drive après upload, ou clonage depuis un repository GitHub public. L'interface Colab détecte automatiquement le format Jupyter et configure l'environnement d'exécution approprié.

L'installation des dépendances s'effectue via la première cellule du notebook, utilisant pip pour installer les bibliothèques spécialisées non incluses dans l'environnement Colab standard. La commande d'installation consolidée `!pip install earthengine-api pandas geopandas openpyxl scikit-learn xgboost folium fpdf2 -q` télécharge et installe automatiquement toutes les dépendances avec leurs versions compatibles.

La configuration du runtime Colab peut être optimisée en sélectionnant un environnement avec GPU pour accélérer l'entraînement des modèles XGBoost. Cette option, accessible via le menu "Runtime > Change runtime type", active l'accès aux GPU Tesla T4 disponibles gratuitement avec certaines limitations d'usage. L'activation du GPU peut réduire significativement les temps d'entraînement pour les jeux de données volumineux.

### Authentification Google Earth Engine

L'authentification Google Earth Engine constitue une étape critique nécessitant une attention particulière pour éviter les erreurs courantes. Le processus d'authentification établit une connexion sécurisée entre l'environnement Colab et les serveurs GEE, permettant l'accès aux données satellites et aux capacités de traitement distribuées.

La procédure d'authentification débute par l'exécution de la cellule contenant `ee.Authenticate()`. Cette commande génère une URL d'authentification unique que l'utilisateur doit ouvrir dans un nouvel onglet. L'URL redirige vers une page Google demandant l'autorisation d'accès aux services Earth Engine pour l'application Colab. L'acceptation de cette autorisation génère un code d'authentification à copier dans l'interface Colab.

Une fois le code d'authentification saisi, l'exécution de `ee.Initialize()` établit la connexion avec les serveurs GEE et vérifie les permissions d'accès. Un message de confirmation indique le succès de l'authentification et la disponibilité des services Earth Engine. Cette authentification reste valide pour la durée de la session Colab et doit être renouvelée à chaque nouvelle session.

Les erreurs d'authentification les plus courantes incluent l'expiration du code d'authentification (validité de 10 minutes), l'utilisation d'un compte Google différent de celui approuvé pour Earth Engine, ou des restrictions réseau bloquant l'accès aux services Google. La résolution de ces erreurs nécessite généralement de répéter la procédure d'authentification avec attention aux détails mentionnés.

### Préparation des Données d'Entrée

La préparation des données d'entrée suit un format standardisé garantissant la compatibilité avec l'ensemble du pipeline de traitement. Le système accepte les données sous forme de fichier Excel (.xlsx) contenant les informations géographiques et attributaires des polygones à classifier.

Le format de données requis inclut trois colonnes essentielles : `geom_json` contenant la géométrie de chaque polygone au format GeoJSON, `culture` spécifiant le type de culture réel pour l'entraînement et la validation, et `year_creation` indiquant l'année de création ou d'observation du polygone. Des colonnes supplémentaires peuvent être incluses et seront préservées dans les résultats d'export.

La colonne `geom_json` doit contenir des géométries valides au format GeoJSON string, supportant les types Polygon et MultiPolygon. Les coordonnées doivent être exprimées en degrés décimaux (WGS84, EPSG:4326) avec une précision suffisante pour la résolution Sentinel-2 (généralement 6 décimales). La validation automatique des géométries détecte et signale les erreurs de format, les géométries invalides, ou les coordonnées aberrantes.

La colonne `culture` accepte des valeurs textuelles décrivant les types de cultures. Les valeurs supportées incluent "palmier à huile", "Cocoa", "Café Arabica", "Café Robusta", "Céréales", "Zone Tampon", et "Autre". Le système gère automatiquement les variations de casse et les espaces supplémentaires, mais une normalisation préalable des données améliore la robustesse du traitement.

### Configuration des Paramètres d'Analyse

La configuration des paramètres d'analyse permet d'adapter le système aux spécificités de chaque projet tout en conservant des valeurs par défaut optimisées pour la plupart des cas d'usage. Ces paramètres contrôlent les aspects critiques du traitement : périodes temporelles, seuils de qualité, et hyperparamètres des modèles.

Les paramètres temporels définissent les périodes d'analyse pour la détection des changements avant/après 2020. Les valeurs par défaut utilisent 2018-2019 pour la période "avant" et 2021-2022 pour la période "après", évitant l'année 2020 marquée par des perturbations liées à la pandémie COVID-19. Ces périodes peuvent être ajustées selon les besoins spécifiques du projet, en respectant la disponibilité des données Sentinel-2 (depuis juin 2015).

Les seuils de qualité des images incluent le pourcentage maximum de couverture nuageuse (20% par défaut) et les critères de sélection des images. Le seuil de couverture nuageuse représente un compromis entre la qualité des données et la disponibilité temporelle, particulièrement important dans les régions tropicales où la couverture nuageuse est fréquente. Des seuils plus stricts (10%) améliorent la qualité au détriment de la disponibilité temporelle.

Les hyperparamètres des modèles de machine learning sont pré-configurés selon les meilleures pratiques identifiées lors du développement. Random Forest utilise 100 arbres avec une profondeur maximale adaptative et une pondération équilibrée des classes. XGBoost emploie 100 itérations avec régularisation L1/L2 et early stopping pour prévenir le surajustement. Ces paramètres peuvent être ajustés pour des cas d'usage spécifiques nécessitant des optimisations particulières.

### Validation de l'Installation

La validation de l'installation s'effectue via une série de tests automatisés vérifiant le bon fonctionnement de chaque composant du système. Cette procédure de validation détecte proactivement les problèmes de configuration et guide l'utilisateur vers les corrections nécessaires.

Le test d'authentification Google Earth Engine vérifie la connectivité avec les serveurs GEE et l'accès aux collections de données Sentinel-2. Ce test charge une image de test et extrait des statistiques basiques pour confirmer le bon fonctionnement de l'API. Un échec à cette étape indique généralement un problème d'authentification ou de permissions d'accès.

Le test des bibliothèques de machine learning vérifie l'installation et la compatibilité des versions de Scikit-learn et XGBoost. Ce test entraîne des modèles miniatures sur des données synthétiques et valide les fonctionnalités de prédiction et de calibration des probabilités. Les erreurs détectées peuvent indiquer des conflits de versions ou des installations incomplètes.

Le test de l'interface utilisateur valide le fonctionnement des composants React et la génération des visualisations. Ce test crée des graphiques de test et vérifie la responsivité de l'interface sur différentes tailles d'écran. Les problèmes détectés peuvent nécessiter la mise à jour du navigateur ou l'activation de JavaScript.

Le test d'export vérifie la génération des fichiers de résultats dans les trois formats supportés (CSV, GeoJSON, PDF). Ce test utilise des données de démonstration pour créer des exports complets et valider leur intégrité. Les erreurs d'export peuvent indiquer des problèmes de permissions de fichiers ou de mémoire disponible.

### Résolution des Problèmes Courants

La résolution des problèmes courants s'appuie sur une base de connaissances développée lors des phases de test et de validation du système. Cette section fournit des solutions détaillées pour les erreurs les plus fréquemment rencontrées, permettant aux utilisateurs de résoudre autonomement la plupart des difficultés.

Les erreurs d'authentification Google Earth Engine représentent la catégorie de problèmes la plus fréquente. La solution standard consiste à vérifier l'approbation du compte GEE, renouveler l'authentification avec le bon compte Google, et s'assurer de la connectivité réseau. Les erreurs persistantes peuvent nécessiter la création d'un nouveau projet GEE ou la demande d'assistance au support Google Earth Engine.

Les erreurs de mémoire dans Google Colab surviennent lors du traitement de jeux de données volumineux dépassant les limites de RAM disponibles (12-25 GB selon le type de runtime). Les solutions incluent la réduction de la taille des échantillons, le traitement par lots, ou l'upgrade vers Colab Pro pour accéder à des ressources étendues. L'optimisation du code peut également réduire l'empreinte mémoire.

Les erreurs de format de données résultent généralement de géométries invalides ou de colonnes manquantes dans le fichier d'entrée. La validation préalable des données avec des outils SIG (QGIS, ArcGIS) permet de détecter et corriger ces problèmes. Le système fournit des messages d'erreur détaillés indiquant les lignes et colonnes problématiques pour faciliter la correction.

Les problèmes de performance peuvent survenir avec des jeux de données très volumineux ou des configurations sous-optimales. L'activation du GPU Colab, l'optimisation des paramètres de modèles, et la parallélisation des calculs constituent les principales approches d'optimisation. Le monitoring des ressources via l'interface Colab aide à identifier les goulots d'étranglement.


## 5. Guide d'Utilisation {#utilisation}

### Workflow Général d'Utilisation

Le workflow d'utilisation du système de classification des cultures agricoles suit une séquence logique d'étapes conçue pour maximiser l'efficacité et minimiser les risques d'erreur. Cette approche structurée guide l'utilisateur depuis la préparation initiale des données jusqu'à l'interprétation des résultats finaux, en passant par l'exécution des analyses et la validation des outputs.

La première phase du workflow concerne la préparation et la validation des données d'entrée. Cette étape critique détermine largement la qualité des résultats finaux et nécessite une attention particulière aux détails de formatage et de cohérence des données. L'utilisateur doit s'assurer que les géométries des polygones sont valides, que les attributs de culture sont correctement renseignés, et que les coordonnées géographiques correspondent effectivement aux zones d'intérêt.

La deuxième phase implique l'exécution séquentielle des cellules du notebook Colab, en respectant l'ordre prescrit pour éviter les erreurs de dépendances. Chaque cellule produit des outputs informatifs permettant de suivre la progression de l'analyse et de détecter d'éventuels problèmes. L'utilisateur doit porter une attention particulière aux messages d'erreur ou d'avertissement qui peuvent indiquer des problèmes nécessitant une intervention.

La troisième phase consiste en l'analyse et l'interprétation des résultats via l'interface dashboard interactive. Cette phase permet d'explorer les résultats sous différents angles, d'identifier les patterns significatifs, et d'évaluer la qualité globale de la classification. L'interface fournit des outils d'analyse visuelle et statistique facilitant cette interprétation.

La phase finale concerne l'export et la sauvegarde des résultats dans les formats appropriés selon les besoins d'utilisation ultérieure. Cette étape garantit la persistance des résultats et leur intégration dans des workflows d'analyse existants ou des systèmes d'information géographique.

### Préparation des Données d'Entrée

La préparation des données d'entrée constitue une étape fondamentale dont la qualité d'exécution influence directement la fiabilité des résultats de classification. Cette préparation nécessite une compréhension approfondie des exigences de format et des bonnes pratiques de structuration des données géospatiales.

Le format de fichier recommandé est Excel (.xlsx) pour sa compatibilité universelle et sa facilité de manipulation. Le fichier doit contenir au minimum trois colonnes essentielles : `geom_json` pour les géométries, `culture` pour les types de cultures, et `year_creation` pour les informations temporelles. Des colonnes supplémentaires peuvent être incluses pour enrichir l'analyse ou faciliter la traçabilité des données.

La colonne `geom_json` doit contenir des géométries au format GeoJSON string, respectant rigoureusement la spécification RFC 7946. Les géométries supportées incluent les types "Polygon" et "MultiPolygon", permettant de gérer aussi bien les parcelles simples que les parcelles fragmentées. Les coordonnées doivent être exprimées en degrés décimaux dans le système WGS84 (EPSG:4326), avec une précision d'au moins 6 décimales pour garantir une résolution compatible avec Sentinel-2.

Un exemple de géométrie valide : `{"type": "Polygon", "coordinates": [[[-3.2, 5.5], [-3.1, 5.5], [-3.1, 5.6], [-3.2, 5.6], [-3.2, 5.5]]]}`. Cette géométrie définit un rectangle simple avec des coordonnées en longitude/latitude. Pour les MultiPolygons, la structure devient : `{"type": "MultiPolygon", "coordinates": [[[polygon1_coords]], [[polygon2_coords]]]}`.

La validation des géométries peut être effectuée préalablement avec des outils SIG comme QGIS. Les erreurs courantes incluent les polygones auto-intersectants, les coordonnées inversées (latitude/longitude), les géométries dégénérées (surface nulle), ou les coordonnées aberrantes (hors des limites géographiques plausibles). Le système détecte automatiquement ces erreurs mais leur correction préalable améliore l'efficacité du traitement.

La colonne `culture` accepte des valeurs textuelles décrivant les types de cultures observées. Les valeurs recommandées incluent "palmier à huile", "Cocoa", "Café Arabica", "Café Robusta", "Céréales", "Zone Tampon", et "Autre" pour les cas non classifiés. La cohérence de la nomenclature est cruciale : "palmier à huile" et "Palmier à huile" seront traités comme des classes différentes. Une normalisation préalable (casse, espaces, accents) améliore la robustesse du traitement.

La colonne `year_creation` doit contenir des années au format numérique (ex: 2018, 2019, 2020). Cette information permet d'analyser les tendances temporelles et d'adapter les périodes d'extraction des données satellites. Les valeurs manquantes sont automatiquement remplacées par la médiane des années disponibles, mais une saisie complète améliore la précision de l'analyse temporelle.

### Exécution du Pipeline d'Analyse

L'exécution du pipeline d'analyse suit une séquence prédéfinie de cellules dans le notebook Colab, chacune accomplissant une fonction spécifique dans le processus global de classification. Cette approche modulaire permet un contrôle fin du processus et facilite le débogage en cas de problème.

La première étape consiste en l'installation des dépendances via la cellule dédiée. Cette opération télécharge et installe automatiquement toutes les bibliothèques nécessaires avec leurs versions compatibles. L'exécution prend généralement 2-3 minutes et produit des messages de progression indiquant l'avancement de l'installation. Les erreurs à cette étape indiquent généralement des problèmes de connectivité réseau ou de permissions Colab.

L'authentification Google Earth Engine constitue la deuxième étape critique. L'exécution de la cellule d'authentification génère une URL unique à ouvrir dans un nouvel onglet. Cette URL redirige vers une page d'autorisation Google où l'utilisateur doit accepter les permissions demandées. Le code d'authentification généré doit être copié dans l'interface Colab pour compléter la procédure. Un message de confirmation indique le succès de l'authentification.

Le chargement des données s'effectue via l'interface d'upload de Colab. L'utilisateur doit cliquer sur le bouton "Choisir les fichiers" et sélectionner le fichier Excel préparé. L'upload peut prendre quelques secondes à quelques minutes selon la taille du fichier et la vitesse de connexion. Une fois l'upload terminé, le système affiche automatiquement un aperçu des données chargées avec des statistiques descriptives.

L'extraction des données Sentinel-2 représente l'étape la plus longue du processus, pouvant prendre de quelques minutes à plusieurs heures selon le nombre de polygones et la complexité des géométries. Le système affiche des messages de progression indiquant le nombre de polygones traités. Cette étape utilise intensivement l'API Google Earth Engine et peut être limitée par les quotas d'utilisation en cas de volumes très importants.

L'entraînement des modèles de machine learning s'exécute automatiquement une fois les données extraites. Cette étape comprend la préparation des features, la division train/test, l'entraînement des modèles Random Forest et XGBoost, et la calibration des probabilités. Les métriques de performance sont affichées en temps réel, permettant d'évaluer la qualité des modèles entraînés.

### Utilisation de l'Interface Dashboard

L'interface dashboard constitue le point d'entrée principal pour l'exploration et l'analyse des résultats de classification. Cette interface moderne et intuitive organise l'information en sections logiques facilitant la navigation et l'interprétation des données.

La page d'accueil du dashboard présente une vue d'ensemble des résultats avec quatre métriques clés affichées sous forme de cartes : nombre total de polygones analysés, nombre de polygones avec haute confiance, confiance moyenne du système, et précision globale du modèle. Ces métriques fournissent une évaluation rapide de la qualité et de la fiabilité de l'analyse effectuée.

L'onglet "Résultats" présente une liste détaillée de tous les polygones classifiés avec leurs attributs principaux : culture réelle, culture prédite, probabilité de prédiction, niveau de confiance, et action recommandée. Cette vue tabulaire permet un tri et un filtrage des résultats selon différents critères. Chaque ligne est cliquable pour afficher des détails supplémentaires dans un panneau dédié.

L'onglet "Carte Interactive" offre une visualisation géospatiale des résultats avec un code couleur basé sur les niveaux de confiance. Les polygones sont colorés selon une échelle de vert (haute confiance) à rouge (faible confiance), permettant une identification visuelle immédiate des zones problématiques. Les popups interactifs affichent les détails de chaque polygone au clic, incluant toutes les informations de classification et les recommandations d'action.

L'onglet "Analyses" présente des graphiques statistiques détaillés : distribution des cultures prédites, répartition des niveaux de confiance, matrice de confusion, et histogrammes des probabilités. Ces visualisations facilitent l'identification de patterns dans les données et l'évaluation de la performance des modèles selon différentes perspectives.

La navigation entre les onglets est fluide et préserve l'état des sélections et filtres appliqués. L'interface est entièrement responsive et s'adapte automatiquement aux différentes tailles d'écran, garantissant une expérience utilisateur optimale sur ordinateurs portables, tablettes et écrans haute résolution.

### Interprétation des Résultats

L'interprétation des résultats nécessite une compréhension approfondie du système de confiance à 5 niveaux et de ses implications pratiques pour l'utilisation des classifications produites. Cette interprétation guide les décisions d'utilisation des résultats et les actions de validation complémentaires nécessaires.

Le niveau de confiance 5 (Très haute confiance) correspond aux prédictions avec une probabilité supérieure à 80%. Ces résultats peuvent être utilisés directement pour la cartographie automatique sans validation supplémentaire. L'analyse statistique montre que ce niveau correspond à une précision effective supérieure à 95%, justifiant une confiance élevée dans ces classifications. Les polygones de niveau 5 représentent généralement des cas "faciles" avec des signatures spectrales distinctives.

Le niveau de confiance 4 (Confiance élevée) couvre les probabilités entre 60% et 80%. Ces résultats sont recommandés pour la cartographie automatique avec des vérifications ponctuelles sur un échantillon représentatif. La précision effective de ce niveau dépasse généralement 85%, offrant un bon compromis entre automatisation et contrôle qualité. Les vérifications peuvent se concentrer sur les bordures de seuil (probabilités proches de 60%).

Le niveau de confiance 3 (Confiance moyenne) correspond aux probabilités entre 40% et 60%. Ces résultats nécessitent une vérification visuelle systématique avant utilisation opérationnelle. Ce niveau identifie souvent des zones de transition entre cultures ou des parcelles avec des caractéristiques spectrales ambiguës. L'expertise humaine apporte une valeur ajoutée significative pour ces cas.

Les niveaux de confiance 1 et 2 (Faible et très faible confiance) identifient les cas problématiques nécessitant une investigation approfondie. Ces résultats peuvent indiquer des erreurs dans les données d'entrée, des conditions d'acquisition défavorables (nuages, ombres), ou des types de cultures non représentés dans les données d'entraînement. Une vérification sur le terrain est recommandée pour ces cas.

L'analyse des erreurs de classification fournit des insights précieux sur les limitations du système et les axes d'amélioration. Les confusions les plus fréquentes (ex: palmier à huile vs cacao jeune) indiquent des défis intrinsèques liés aux similarités spectrales. Ces informations guident les stratégies d'amélioration : collecte de données supplémentaires, intégration de features temporelles, ou utilisation d'indices de végétation spécialisés.

### Export et Sauvegarde des Résultats

L'export des résultats s'effectue via trois formats complémentaires optimisés pour différents usages : CSV pour l'analyse statistique, GeoJSON pour l'intégration SIG, et PDF pour la communication et l'archivage. Cette diversité de formats garantit la compatibilité avec l'ensemble des workflows d'analyse existants.

L'export CSV génère un fichier structuré contenant l'ensemble des résultats de classification avec leurs métadonnées associées. Le fichier inclut les colonnes suivantes : identifiant du polygone, culture réelle, culture prédite, probabilité de prédiction, niveau de confiance, action recommandée, année de création, et caractéristiques spectrales utilisées. Ce format facilite l'analyse statistique avec des outils comme Excel, R, Python, ou des logiciels statistiques spécialisés.

L'export GeoJSON préserve l'information géospatiale complète en intégrant les géométries des polygones avec leurs attributs de classification. Ce format respecte les standards OGC et assure une compatibilité maximale avec les logiciels SIG (QGIS, ArcGIS, MapInfo) et les plateformes web de cartographie. Les attributs de classification sont stockés comme propriétés des features géographiques, permettant une symbolisation et une analyse spatiale avancées.

L'export PDF génère automatiquement un rapport exécutif professionnel incluant une synthèse des résultats, des métriques de performance, des recommandations d'utilisation, et des visualisations clés. Ce format facilite la communication des résultats aux décideurs et parties prenantes non techniques. Le rapport inclut également une section méthodologique résumant les approches utilisées et les limitations à considérer.

La sauvegarde des résultats dans Google Drive s'effectue automatiquement lors de l'export, garantissant la persistance des données au-delà de la session Colab. Les fichiers sont organisés dans un dossier dédié avec une nomenclature incluant la date et l'heure d'exécution pour faciliter la traçabilité. Cette organisation permet de conserver un historique des analyses et de comparer les résultats de différentes configurations ou jeux de données.

### Bonnes Pratiques d'Utilisation

L'adoption de bonnes pratiques d'utilisation maximise la qualité des résultats et minimise les risques d'erreur ou de mauvaise interprétation. Ces pratiques, développées lors des phases de test et de validation, reflètent l'expérience accumulée avec le système et les retours d'utilisateurs pilotes.

La validation préalable des données d'entrée constitue la première bonne pratique essentielle. Cette validation inclut la vérification de la cohérence géographique (polygones dans la zone d'étude), la validation des géométries (absence d'auto-intersections, surfaces non nulles), et la cohérence des attributs (nomenclature standardisée des cultures). L'utilisation d'outils SIG pour cette validation préalable évite de nombreux problèmes lors de l'exécution.

La gestion des sessions Colab nécessite une attention particulière aux limitations de temps et de ressources. Les sessions Colab ont une durée de vie limitée (12 heures maximum) et peuvent être interrompues en cas d'inactivité prolongée. Pour les analyses longues, il est recommandé de sauvegarder régulièrement les résultats intermédiaires et de maintenir une activité minimale dans l'interface pour éviter les déconnexions.

L'interprétation des résultats doit toujours considérer le contexte géographique et temporel de l'analyse. Les performances des modèles peuvent varier selon les régions (différences climatiques, pratiques agricoles), les saisons (phénologie des cultures), et les années (conditions météorologiques exceptionnelles). Une analyse critique des résultats en fonction de ces facteurs améliore la fiabilité de l'interprétation.

La documentation des analyses effectuées facilite la reproductibilité et le partage des résultats. Cette documentation doit inclure les paramètres utilisés, les sources de données, les éventuelles modifications apportées au code, et les observations particulières relevées lors de l'exécution. Cette traçabilité est particulièrement importante pour les projets de recherche ou les applications opérationnelles nécessitant une validation externe.

