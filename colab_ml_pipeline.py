
# -*- coding: utf-8 -*-
"""Ceci est un script Python pour Google Colab pour le pipeline ML.

Partie 1: Installation des dépendances et authentification Google Earth Engine.
Partie 2: Simulation de l'acquisition des données Sentinel-2 et calcul NDVI.
"""

# 1. Installation des dépendances (à exécuter une seule fois dans Colab)
# !pip install earthengine-api pandas geopandas openpyxl scikit-learn xgboost folium

# 2. Importation des bibliothèques
import pandas as pd
import geopandas as gpd
import json
import numpy as np
from shapely.geometry import shape

# 3. Chargement des données d'entrée
print("Chargement des données d'entrée...")
excel_file_path = '/home/ubuntu/upload/bd_ml.xlsx'
df = pd.read_excel(excel_file_path)

# Convertir la colonne 'geom_json' en objets géométriques (pour la simulation, pas pour GEE)
print("Conversion des géométries GeoJSON...")
def parse_geojson_to_geometry(geojson_str):
    try:
        geojson_dict = json.loads(geojson_str)
        return shape(geojson_dict)
    except Exception as e:
        print(f"Erreur lors de l'analyse GeoJSON: {e} pour {geojson_str}")
        return None

df['geometry_obj'] = df['geom_json'].apply(parse_geojson_to_geometry)

# Filtrer les lignes où la géométrie n'a pas pu être parsée
df_valid_geom = df.dropna(subset=['geometry_obj'])
print(f"Nombre de polygones valides pour traitement: {len(df_valid_geom)}")

# 4. Simulation de l'extraction des caractéristiques (NDVI et autres)
print("Simulation de l'extraction des caractéristiques Sentinel-2 (NDVI et autres)...")

def generate_synthetic_features(num_rows, prefix):
    data = {
        f'{prefix}_NDVI_mean': np.random.uniform(0.1, 0.9, num_rows),
        f'{prefix}_NDVI_median': np.random.uniform(0.1, 0.9, num_rows),
        f'{prefix}_NDVI_stdDev': np.random.uniform(0.01, 0.1, num_rows),
        f'{prefix}_NDVI_min': np.random.uniform(0.0, 0.8, num_rows),
        f'{prefix}_NDVI_max': np.random.uniform(0.2, 1.0, num_rows),
        f'{prefix}_B2_mean': np.random.uniform(500, 2000, num_rows),
        f'{prefix}_B3_mean': np.random.uniform(500, 2000, num_rows),
        f'{prefix}_B4_mean': np.random.uniform(500, 2000, num_rows),
        f'{prefix}_B8_mean': np.random.uniform(1000, 4000, num_rows),
        f'{prefix}_B11_mean': np.random.uniform(500, 3000, num_rows),
        f'{prefix}_B12_mean': np.random.uniform(500, 3000, num_rows),
    }
    return pd.DataFrame(data)

num_polygons = len(df_valid_geom)

df_features_before = generate_synthetic_features(num_polygons, 'before')
df_features_after = generate_synthetic_features(num_polygons, 'after')

# Concaténer les caractéristiques simulées avec le DataFrame original
df_processed = pd.concat([df_valid_geom.reset_index(drop=True), df_features_before, df_features_after], axis=1)

print("Extraction des caractéristiques simulée terminée. Aperçu des données traitées:")
print(df_processed.head().to_markdown(index=False))

# Sauvegarder les données traitées (optionnel, pour inspection)
df_processed.to_csv('processed_data_simulated.csv', index=False)
print("Données traitées simulées sauvegardées dans 'processed_data_simulated.csv'")

# 5. Préparation des données pour le Machine Learning
print("Préparation des données pour le Machine Learning...")

# Définir les caractéristiques (features) et la variable cible (target)
# Les caractéristiques incluent les données simulées et l'année de création
features_cols = [col for col in df_processed.columns if col.startswith(('before_', 'after_'))]
features_cols.append('year_creation') # Ajouter l'année de création comme caractéristique

# Gérer les valeurs manquantes dans 'year_creation' (par exemple, remplacer par la médiane ou la moyenne)
df_processed['year_creation'] = df_processed['year_creation'].fillna(df_processed['year_creation'].median())

# Gérer les valeurs manquantes dans 'culture' en les supprimant
df_processed_cleaned = df_processed.dropna(subset=['culture']).copy()

X = df_processed_cleaned[features_cols]
y = df_processed_cleaned['culture']

# Encodage de la variable cible si elle est textuelle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()

# Encodage de la variable cible sur l'ensemble complet
y_encoded = le.fit_transform(y)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


print(f"Caractéristiques (X) shape: {X.shape}")
print(f"Cible (y) shape: {y.shape}")
print(f"Classes encodées: {le.classes_}")

print("Données divisées en ensembles d'entraînement et de test.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_encoded.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_encoded.shape}")

# 6. Implémentation des modèles ML (Random Forest + XGBoost) avec calibration
print("Implémentation des modèles ML (Random Forest + XGBoost) avec calibration...")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# Random Forest
print("Entraînement du modèle Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train_encoded)
rf_preds = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)
print(f"Accuracy Random Forest: {accuracy_score(y_test_encoded, rf_preds):.4f}")
print("Rapport de classification Random Forest:\n", classification_report(y_test_encoded, rf_preds, labels=np.unique(y_test_encoded), target_names=le.classes_[np.unique(y_test_encoded)], zero_division=0))
print("Entraînement du modèle XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1)
# Remapper les labels pour XGBoost pour qu'ils soient contigus et commencent à 00
# Si certaines classes sont absentes du jeu d\'entraînement, LabelEncoder peut produire des labels non contigus.
# Nous allons utiliser pd.factorize pour remapper les labels.
y_train_xgb = pd.factorize(y_train_encoded)[0]
y_test_xgb = pd.factorize(y_test_encoded)[0]
xgb_model.fit(X_train, y_train_xgb)

xgb_preds = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)
print(f"Accuracy XGBoost: {accuracy_score(y_test_encoded, xgb_preds):.4f}")
print("Rapport de classification XGBoost:\n", classification_report(y_test_encoded, xgb_preds, labels=np.unique(y_test_encoded), target_names=le.classes_[np.unique(y_test_encoded)], zero_division=0))

# Calibration des probabilités (pour Random Forest, par exemple)
print("Calibration des probabilités...")
calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv='prefit')
calibrated_rf.fit(X_train, y_train_encoded)
calibrated_rf_proba = calibrated_rf.predict_proba(X_test)

print("Probabilités calibrées générées.")

# 7. Système de confiance à 5 niveaux avec actions recommandées
print("Développement du système de confiance à 5 niveaux...")

def get_confidence_level(probabilities):
    max_proba = np.max(probabilities)
    if max_proba >= 0.8:
        return 5, "Très haute confiance: Peut être utilisé pour la cartographie automatique."
    elif max_proba >= 0.6:
        return 4, "Confiance élevée: Cartographie automatique recommandée, vérification ponctuelle possible."
    elif max_proba >= 0.4:
        return 3, "Confiance moyenne: Vérification visuelle recommandée si possible."
    elif max_proba >= 0.2:
        return 2, "Faible confiance: Nécessite une vérification sur le terrain ou une analyse visuelle approfondie."
    else:
        return 1, "Très faible confiance: Vérification sur le terrain indispensable."

# Appliquer le système de confiance aux probabilités calibrées du RF (ou du modèle combiné)
confidence_results = [get_confidence_level(p) for p in calibrated_rf_proba]
confidence_level = [res[0] for res in confidence_results]
recommended_action = [res[1] for res in confidence_results]

df_results = df_processed_cleaned.loc[X_test.index].copy()
df_results['predicted_culture'] = le.inverse_transform(calibrated_rf.predict(X_test))
df_results['predicted_proba'] = np.max(calibrated_rf_proba, axis=1)
df_results['confidence_level'] = confidence_level
df_results['recommended_action'] = recommended_action

print("Aperçu des résultats avec niveaux de confiance:")
print(df_results[['culture', 'predicted_culture', 'predicted_proba', 'confidence_level', 'recommended_action']].head().to_markdown(index=False))

# 8. Création du dashboard temps réel avec cartes colorées (ébauche pour Colab)
print("Ébauche du dashboard temps réel avec cartes colorées...")

import folium

# Créer une carte centrée sur une zone moyenne des polygones
# Utiliser les coordonnées du premier polygone valide comme centre initial
if not df_results.empty:
    first_geom = df_results['geometry_obj'].iloc[0]
    if first_geom:
        map_center = [first_geom.centroid.y, first_geom.centroid.x]
    else:
        map_center = [0, 0] # Fallback
else:
    map_center = [0, 0] # Fallback

m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter') # Dark mode

# Définir une palette de couleurs pour les niveaux de confiance
colors = {
    1: 'red',    # Très faible confiance
    2: 'orange', # Faible confiance
    3: 'yellow', # Confiance moyenne
    4: 'lightgreen', # Confiance élevée
    5: 'green'   # Très haute confiance
}

# Ajouter les polygones à la carte avec des couleurs basées sur le niveau de confiance
for idx, row in df_results.iterrows():
    if row['geometry_obj']:
        # Folium gère les objets shapely directement
        # Pour MultiPolygon, il faut itérer sur les polygones individuels
        if row['geometry_obj'].geom_type == 'MultiPolygon':
            for poly in row['geometry_obj'].geoms:
                folium.GeoJson(
                    poly.__geo_interface__,
                    style_function=lambda x, color=colors.get(row['confidence_level'], 'gray'): {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=f"Culture: {row['predicted_culture']}<br>Proba: {row['predicted_proba']:.2f}<br>Confiance: {row['confidence_level']}<br>Action: {row['recommended_action']}"
                ).add_to(m)
        else:
            folium.GeoJson(
                row['geometry_obj'].__geo_interface__,
                style_function=lambda x, color=colors.get(row['confidence_level'], 'gray'): {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                tooltip=f"Culture: {row['predicted_culture']}<br>Proba: {row['predicted_proba']:.2f}<br>Confiance: {row['confidence_level']}<br>Action: {row['recommended_action']}"
            ).add_to(m)

# Sauvegarder la carte HTML (peut être affichée directement dans Colab)
map_output_path = 'culture_classification_map.html'
m.save(map_output_path)
print(f"Carte interactive sauvegardée dans '{map_output_path}'")

# 9. Implémentation des exports CSV/PDF/GeoJSON
print("Implémentation des exports CSV/PDF/GeoJSON...")

# Export CSV
csv_output_path = 'classification_results.csv'
df_results.to_csv(csv_output_path, index=False)
print(f"Résultats exportés au format CSV: '{csv_output_path}'")

# Export GeoJSON
# Créer un GeoDataFrame à partir de df_results
gdf_results = gpd.GeoDataFrame(
    df_results,
    geometry='geometry_obj',
    crs="EPSG:4326" # WGS84, système de coordonnées standard pour les données géographiques
)
geojson_output_path = 'classification_results.geojson'
gdf_results.to_file(geojson_output_path, driver='GeoJSON')
print(f"Résultats exportés au format GeoJSON: '{geojson_output_path}'")

# Export PDF (exemple simple)
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Rapport de Classification des Cultures', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()
pdf.chapter_title('Résumé des Résultats')

summary_text = f"Nombre total de polygones traités: {len(df_results)}\n"
for culture in le.classes_:
    count = (df_results['predicted_culture'] == culture).sum()
    summary_text += f"- {culture}: {count} polygones\n"

summary_text += "\n\nDétail des résultats (premières lignes):\n"
summary_text += df_results[['culture', 'predicted_culture', 'predicted_proba', 'confidence_level', 'recommended_action']].head().to_string(index=False)

pdf.chapter_body(summary_text)

pdf_output_path = 'classification_report.pdf'
pdf.output(pdf_output_path)
print(f"Rapport exporté au format PDF: '{pdf_output_path}'")

print("Pipeline ML complet (simulation GEE) terminé.")


