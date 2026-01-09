import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
import os

file_path = r'E:\Landslide_data\My_excel_data.xlsx' #Data path
data = pd.read_excel(file_path, engine='openpyxl')


def categorize_aspect(angle):
    if angle == -1 or angle == 0:
        return 'Flat'
    elif angle >= 337.5 or angle <= 22.5:
        return 'North'
    elif 22.5 < angle <= 67.5:
        return 'Northeast'
    elif 67.5 < angle <= 112.5:
        return 'East'
    elif 112.5 < angle <= 157.5:
        return 'Southeast'
    elif 157.5 < angle <= 202.5:
        return 'South'
    elif 202.5 < angle <= 247.5:
        return 'Southwest'
    elif 247.5 < angle <= 292.5:
        return 'West'
    elif 292.5 < angle < 337.5:
        return 'Northwest'

data['Aspect_cat'] = data['Aspect'].apply(categorize_aspect)


lulc_map = {
    1: 1, 2: 2, 4: 4, 5: 5, 7: 7, 8: 8, 11: 11
}
data['LULC_cat'] = data['LULC'].round().map(lulc_map)


geology_map = {i: i for i in range(1, 8)}
geology_map.update({i: i for i in range(11, 18)})
data['Geology_cat'] = data['Geology'].round().map(geology_map)


aspect_dummies = pd.get_dummies(data['Aspect_cat'], prefix='Aspect').astype(int)
lulc_dummies   = pd.get_dummies(data['LULC_cat'], prefix='LULC').astype(int)
geology_dummies= pd.get_dummies(data['Geology_cat'], prefix='Geology').astype(int)

data_encoded = pd.concat([data, aspect_dummies, lulc_dummies, geology_dummies], axis=1)
data_encoded.drop(['Aspect','Aspect_cat','LULC','LULC_cat','Geology','Geology_cat'], axis=1, inplace=True)

# Target & features
y = data_encoded['Event']
X = data_encoded.drop('Event', axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


importances = rf_model.feature_importances_
features = X.columns

grouped_importance = {
    'Elevation': 0, 'Slope': 0, 'NDVI': 0, 'Rainfall': 0,
    'Curvature': 0, 'SPI': 0,
    'TWI': 0, 'TRI': 0, 'Aspect': 0, 'LULC': 0, 'Geology': 0,
    'Drainage Density': 0, 'Distance to Roads': 0
}

for f, imp in zip(features, importances):
    if 'Aspect_' in f:
        grouped_importance['Aspect'] += imp
    elif 'LULC_' in f:
        grouped_importance['LULC'] += imp
    elif 'Geology_' in f:
        grouped_importance['Geology'] += imp
    else:
        if f in grouped_importance:
            grouped_importance[f] += imp

plt.figure(figsize=(10,6))
plt.bar(grouped_importance.keys(), grouped_importance.values(), color='skyblue')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance (Grouped)')
plt.xticks(rotation=45)
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


import rasterio
from rasterio.windows import Window

# Aspect reclass
def categorize_aspect_val(val):
    if val == -1 or val == 0: return 0
    elif val >= 337.5 or val <= 22.5: return 1
    elif 22.5 < val <= 67.5: return 2
    elif 67.5 < val <= 112.5: return 3
    elif 112.5 < val <= 157.5: return 4
    elif 157.5 < val <= 202.5: return 5
    elif 202.5 < val <= 247.5: return 6
    elif 247.5 < val <= 292.5: return 7
    elif 292.5 < val < 337.5: return 8
    return 0

lulc_map = {1:1, 2:2, 4:4, 5:5, 7:7, 8:8, 11:11}
geology_map = {i:i for i in range(1,9)}
geology_map.update({i:i for i in range(11,19)})

# Input rasters
raster_names = [
    'Elevation.tif','Slope_angle.tif','NDVI.tif','Rainfall.tif',
    'Curvature.tif','SPI.tif','TWI.tif','TRI.tif',
    'Aspect.tif','LULC.tif','Geology.tif','DD_fff.tif','distance_roadsss.tif'
]

raster_folder = r"E:\Landslide_rahma\Resampled_Reprojected"
ref_path = os.path.join(raster_folder, 'Elevation.tif')

# Open reference raster
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    rows, cols = ref.height, ref.width

# Output raster
out_path = os.path.join(raster_folder, "Landslide_Susceptibility.tif")
meta = ref_meta.copy()
meta.update(dtype=rasterio.float32, count=1, compress='lzw')

rasters = [rasterio.open(os.path.join(raster_folder, r)) for r in raster_names]

tile_size = 512  

with rasterio.open(out_path, 'w', **meta) as dst:
    for i in range(0, rows, tile_size):
        for j in range(0, cols, tile_size):
            win = Window(j, i, min(tile_size, cols-j), min(tile_size, rows-i))

            tile_arrays = [src.read(1, window=win, out_shape=(win.height, win.width)) for src in rasters]
            stacked = np.stack(tile_arrays, axis=-1).reshape(-1, len(rasters))
            temp_df = pd.DataFrame(stacked, columns=[
                'Elevation','Slope','NDVI','Rainfall',
                'Plan Curvature','Profile Curvature','SPI','TWI','TRI',
                'Aspect','LULC','Geology','Drainage Density','Distance to Roads'
            ])

            # Reclassify
            temp_df['Aspect']  = temp_df['Aspect'].apply(categorize_aspect_val).astype(int)
            temp_df['LULC']    = temp_df['LULC'].round().map(lulc_map).fillna(0).astype(int)
            temp_df['Geology'] = temp_df['Geology'].round().map(geology_map).fillna(0).astype(int)

            # One-hot encoding
            aspect_dummies  = pd.get_dummies(temp_df['Aspect'], prefix='Aspect')
            lulc_dummies    = pd.get_dummies(temp_df['LULC'], prefix='LULC')
            geology_dummies = pd.get_dummies(temp_df['Geology'], prefix='Geology')

            temp_df_encoded = pd.concat(
                [temp_df.drop(['Aspect','LULC','Geology'], axis=1),
                 aspect_dummies, lulc_dummies, geology_dummies],
                axis=1
            )

            # Align with training features
            for col in X.columns:
                if col not in temp_df_encoded.columns:
                    temp_df_encoded[col] = 0
            temp_df_encoded = temp_df_encoded[X.columns]

            # Predict
            tile_pred = rf_model.predict_proba(temp_df_encoded)[:, 1]
            tile_pred = tile_pred.reshape(win.height, win.width)

            dst.write(tile_pred.astype(rasterio.float32), 1, window=win)

print(f"Landslide susceptibility map saved at: {out_path}")
