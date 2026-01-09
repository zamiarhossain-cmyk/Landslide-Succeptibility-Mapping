import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
import rasterio
from rasterio.windows import Window
import os


file_path = r'E:\E:\Landslide_succeptibility\_LSM_Data.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')


def categorize_aspect(angle):
    if angle in (-1, 0): return 'Flat'
    elif angle >= 337.5 or angle <= 22.5: return 'North'
    elif 22.5 < angle <= 67.5: return 'Northeast'
    elif 67.5 < angle <= 112.5: return 'East'
    elif 112.5 < angle <= 157.5: return 'Southeast'
    elif 157.5 < angle <= 202.5: return 'South'
    elif 202.5 < angle <= 247.5: return 'Southwest'
    elif 247.5 < angle <= 292.5: return 'West'
    else: return 'Northwest'

data['Aspect_cat'] = data['Aspect'].apply(categorize_aspect)

lulc_map = {1:1, 2:2, 4:4, 5:5, 7:7, 8:8, 11:11}
geo_map  = {i:i for i in range(1,9)}
geo_map.update({i:i for i in range(11,19)})

data['LULC_cat']    = data['LULC'].round().map(lulc_map)
data['Geology_cat'] = data['Geology'].round().map(geo_map)

data = pd.concat([
    data.drop(['Aspect','LULC','Geology'], axis=1),
    pd.get_dummies(data['Aspect_cat'], prefix='Aspect'),
    pd.get_dummies(data['LULC_cat'], prefix='LULC'),
    pd.get_dummies(data['Geology_cat'], prefix='Geology')
], axis=1).drop(['Aspect_cat','LULC_cat','Geology_cat'], axis=1)


y = data['Event']
X = data.drop('Event', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


grouped = {
    'Elevation':0,'Slope':0,'Curvature':0,'NDVI':0,'Rainfall':0,
    'SPI':0,'TWI':0,'TRI':0,'Aspect':0,'LULC':0,'Geology':0,
    'Drainage Density':0,'Distance to Roads':0
}

for f, imp in zip(X.columns, rf.feature_importances_):
    if 'Aspect_' in f: grouped['Aspect'] += imp
    elif 'LULC_' in f: grouped['LULC'] += imp
    elif 'Geology_' in f: grouped['Geology'] += imp
    elif f in grouped: grouped[f] += imp

plt.bar(grouped.keys(), grouped.values())
plt.xticks(rotation=45)
plt.title("Grouped Feature Importance")
plt.show()


def aspect_reclass(v):
    if v in (-1,0): return 0
    elif v >= 337.5 or v <= 22.5: return 1
    elif 22.5 < v <= 67.5: return 2
    elif 67.5 < v <= 112.5: return 3
    elif 112.5 < v <= 157.5: return 4
    elif 157.5 < v <= 202.5: return 5
    elif 202.5 < v <= 247.5: return 6
    elif 247.5 < v <= 292.5: return 7
    else: return 8

raster_folder = r"E:\Landslide_succeptibility\Resampled_Reprojected" #Location of the rasters
raster_names = [
    'Elevation.tif','Slope_angle.tif','Curvature.tif','NDVI.tif','Rainfall.tif',
    'SPI.tif','TWI.tif','TRI.tif','Aspect.tif','LULC.tif','Geology.tif',
    'Drainage_density.tif','distance_roads.tif'
]

rasters = [rasterio.open(os.path.join(raster_folder, r)) for r in raster_names]
ref = rasters[0]

meta = ref.meta.copy()
meta.update(dtype='float32', count=1)

out_path = os.path.join(raster_folder, "Landslide_Susceptibility.tif")

with rasterio.open(out_path, 'w', **meta) as dst:
    for i in range(0, ref.height, 512):
        for j in range(0, ref.width, 512):
            win = Window(j, i, min(512, ref.width-j), min(512, ref.height-i))
            arr = np.stack([r.read(1, window=win) for r in rasters], axis=-1)
            df = pd.DataFrame(arr.reshape(-1, arr.shape[-1]), columns=X.columns[:len(rasters)])

            df['Aspect']  = df['Aspect'].apply(aspect_reclass)
            df['LULC']    = df['LULC'].round().map(lulc_map).fillna(0)
            df['Geology'] = df['Geology'].round().map(geo_map).fillna(0)

            df = pd.get_dummies(df, columns=['Aspect','LULC','Geology'])
            for c in X.columns:
                if c not in df: df[c] = 0
            df = df[X.columns]

            pred = rf.predict_proba(df)[:,1].reshape(win.height, win.width)
            dst.write(pred.astype('float32'), 1, window=win)

print("Susceptibility map saved.")
