# Taller-1-Deep-Learning

Estructura y flujo de trabajo del laboratorio de Deep Learning para clasificacion de deterioro cognitivo.

## Arquitectura del repositorio

Basada en la seccion 5.9 del documento `P_Laboratorio_Documento_DL_2026_01.pdf`:

```text
Taller-1-Deep-Learning/
|-- data/
|   |-- raw/
|   |   '-- 15 atributos R0-R5.sav
|   |-- processed/
|   '-- outputs/
|-- docs/
|-- src/
|   |-- main.py
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- balancing.py
|   |-- bagging_model.py
|   |-- boosting_model.py
|   |-- stacking_model.py
|   |-- evaluation.py
|   |-- notebook_workflow.py
|   |-- visualization.py
|   '-- utils.py
|-- reports/
|   |-- figuras/
|   '-- tablas/
|-- environment.yml
|-- requirements.txt
'-- README.md
```

## Entorno virtual con Conda

1. Crear el entorno desde el archivo YAML:

```powershell
conda env create -f environment.yml
```

2. Activar el entorno:

```powershell
conda activate taller_1_dl
```

## Dataset local (no versionado)

El archivo `15 atributos R0-R5.sav` se mantiene local y no se sube al repositorio.
Esta ignorado en `.gitignore`.

Ubicacion esperada:

- `data/raw/15 atributos R0-R5.sav`

Si no tienes la carpeta `data/raw/`, creala y copia el archivo `.sav` ahi antes de ejecutar `main.py`.

## Ejecucion en VS Code

1. Abrir la carpeta del repositorio en VS Code.
2. Seleccionar el interprete de Python del entorno `taller_1_dl`.
3. Ejecutar los experimentos desde terminal:

```powershell
python src/main.py
```

`main.py` ejecuta dos bloques en cada estrategia seleccionada:

- bloque 1 (alineado al notebook):

	- distribucion de clases por codificacion (`GDS`, `GDS_R1` ... `GDS_R5`)
	- validacion de atributos binarios
	- seleccion de caracteristicas (`VarianceThreshold`, `Chi2`, `RFE`, `Forward Selection`)
	- PCA (varianza explicada acumulada y componentes al 95%)
	- evaluacion de Naive Bayes manual por codificacion

- bloque 2 (requerido por documentos de `docs`):

	- evaluacion de ensambles `Bagging`, `Boosting`, `Stacking`
	- mismo protocolo de validacion para los tres modelos
	- metricas macro (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`)
	- matrices de confusion por modelo

El programa mostrara un menu para elegir estrategia de validacion:

- `1) Stratified CV (up to 5 folds) [~1-3 minutes]`
- `2) LOOCV [~45-180 minutes]`
- `3) Both strategies [~45-180 minutes +]`

La ejecucion guarda resultados por estrategia en carpetas separadas:

- `reports/tablas/stratified_kfold/`
- `reports/figuras/stratified_kfold/`
- `data/outputs/stratified_kfold/run_summary.json`
- `reports/tablas/loocv/`
- `reports/figuras/loocv/`
- `data/outputs/loocv/run_summary.json`

Archivos principales por estrategia:

- `class_distribution_all_codifications.csv` y `class_distribution_all_codifications.png`
- `features_variance_selected.csv`
- `features_chi2_top15_gds.csv`
- `features_chi2_all_gds.csv`
- `features_rfe_logistic.csv`
- `features_rfe_tree.csv`
- `features_forward_knn.csv`
- `pca_explained_variance.csv`
- `pca_components_95.csv`
- `pca_cumulative_variance.png`
- `naive_bayes_metrics_by_codification.csv`
- `naive_bayes_metrics_by_codification.png`
- `naive_bayes_selected_features_by_codification.csv`
- `model_metrics.csv`
- `model_metrics.png`
- `confusion_matrix_bagging.csv` y `confusion_matrix_bagging.png`
- `confusion_matrix_boosting.csv` y `confusion_matrix_boosting.png`
- `confusion_matrix_stacking.csv` y `confusion_matrix_stacking.png`

Tambien genera agregados globales:

- `reports/tablas/strategy_codification_comparison.csv`
- `reports/tablas/strategy_model_comparison.csv`
- `data/outputs/run_summary.json`

`reports/tablas/` se usa para tablas en formato tabular (CSV) y `reports/figuras/` para imagenes (PNG).

`run_summary.json` guarda rutas relativas al repositorio para que los resultados sean portables y aptos para subir a Git.
