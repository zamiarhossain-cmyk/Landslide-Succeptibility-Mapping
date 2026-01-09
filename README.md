# Landslide-Succeptibility-Mapping
Landslide Susceptibility Mapping using Random Forest

This project develops a Random Forest–based landslide susceptibility model using terrain, environmental, and thematic conditioning factors. The trained model is spatially applied to raster datasets to generate a continuous susceptibility map.

Conditioning Factors

Elevation

Slope

Curvature (single combined curvature parameter)

NDVI

Rainfall

SPI, TWI, TRI

Aspect (directional classes)

Land Use / Land Cover

Geology

Drainage Density

Distance to Roads

Methodology

Aspect, LULC, and Geology treated as categorical variables

One-hot encoding applied consistently for both training and raster prediction

Random Forest trained using stratified 70/30 split

Model performance evaluated using ROC–AUC

Feature importance aggregated by physical meaning

Spatial Prediction

Tile-based raster inference (512×512) for memory efficiency

Strict feature alignment ensures no training–prediction mismatch

Output is a continuous probability raster (0–1)

Output

Landslide_Susceptibility.tif
Higher values indicate higher relative susceptibility

Key Note

This model estimates relative spatial susceptibility, not landslide timing or triggering probability.
