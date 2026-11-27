# Vaca_AI

## ¿Cómo obtener y preparar los datos?

Ejecuta los siguientes pasos en orden para construir el conjunto de datos. Se asume que estas en la carpeta raíz del repositorio para correr los comandos.

### 1. Preparar los archivos y el entorno

1. Descarga los datos de Google Drive.
2. Crea las carpetas `Data/vacas/` y `Data/concentrado` en la raíz del repositorio.
3. Copia a `Data/vacas/` los archivos `Producciones de leche por sesion XXXX.xlsx` de `TEC-UAQ-UNAM` y los archivos `.csv` de `DATOS vacas en establo marzo_junio_2025` cuyo nombre sea un número, los demás puedes ignorarlos.
4. Copia a `Data/concentrado` los archivos `Concentrado consumido XXXX.xlsx` de `TEC-UAQ-UNAM`
5. Instala Python y las dependencias del proyecto:

   **Windows (PowerShell)**

   ```powershell
   py -m pip install -r requirements.txt
   ```

   **macOS y Linux**

   ```bash
   python3 -m pip install -r requirements.txt
   ```

**Resultado esperado:** los archivos originales están en `Data/vacas/` y las dependencias están instaladas.

### 2. Unir los datos de las vacas

**Windows**

```powershell
py src/scripts/generate_data.py
```

**macOS y Linux**

```bash
python3 src/scripts/generate_data.py
```

**Resultado esperado:** se genera `Data/dataset_vacas.csv` con los datos combinados. La terminal muestra el número de archivos procesados, observaciones duplicadas y valores nulos.

### 3. Anonimizar los datos

Crea un archivo `.env` en la raíz con `COWSKEY=tu_llave_secreta`, sustituyendo el valor por una llave privada. Conserva la misma llave si necesitas generar los mismos identificadores anónimos.

**Windows**

```powershell
py src/scripts/anonymize_data.py
```

**macOS y Linux**

```bash
python3 src/scripts/anonymize_data.py
```

**Resultado esperado:** se genera `Data/dataset_vacas_anonimo.csv` con los identificadores anonimizados y `Data/mapa_id_vacas.csv` con sus equivalencias.

### 4. Descargar los datos del clima

**Windows**

```powershell
py src/scripts/descargar_clima.py
```

**macOS y Linux**

```bash
python3 src/scripts/descargar_clima.py
```

**Resultado esperado:** se crea `Data/clima/clima_nasa_power_por_hora.csv`, junto con las respuestas originales y un manifiesto en `Data/clima/`. La terminal muestra la cita de la fuente y una fila para la bitácora de auditoría.

### 5. Limpiar los datos

Ejecuta este paso después de anonimizar los datos (paso 3) y descargar el clima (paso 4).

**Windows**

```powershell
py src/DP/clean_data.py
```

**macOS y Linux**

```bash
python3 src/DP/clean_data.py
```

**Resultado esperado:** se generan `Data/dataset_vacas_limpio.csv` y `Data/clima/clima_nasa_power_limpio.csv`. La terminal muestra un resumen de cada paso de la limpieza. El dataset de vacas queda sin nulos y con estos cambios principales:

- Unifica fechas, duraciones e intervalos del CSV y del XLSX, y quita las sesiones repetidas entre ambas fuentes.
- Convierte las alertas de texto (`DI,TD`, `Todos`) en banderas 0/1 por cuarto y las categóricas en dummies. Cada categórica conserva una categoría base (todas sus dummies en 0): `Ordeño` para la acción, `Tanque` para el destino de la leche, `Sin desvío` para la razón de desviación, `xlsx` para la fuente y, para la máquina, su categoría más frecuente (la moda), porque su hash depende de `COWSKEY`.
- Deja fuera las máquinas con menos de 100 registros.
- Pasa a faltante la conductividad en 0 y trata los nulos usando solo la misma visita o el pasado de la misma vaca. Los nulos que quedan (vacas sin historial) pasan a 0. Los valores estimados o rellenados quedan marcados con banderas (`*_faltante`, `*_estimada_*`, `*_sin_dato`).
- Pasa a 0 kg los cuartos con producción nula (verificado contra el total del ordeño) y los marca con `cuarto_inactivo_<cuarto>`: un cuarto dañado viene nulo y sin alertas, mientras que un fallo del robot viene en 0 con alerta, y los dos casos deben seguir distinguiéndose.
- Marca, sin eliminar, los ordeños con 0 kg, los de suma de cuartos distinta del total y los picos de sangre por máquina.
- Excluye las vacas con menos de 30 registros y agrega `bloque_historial`, que cambia tras 45 días de ausencia.
- Deja `sangre_ppm_*`, `estado_mdi` y `contadores_monitorizacion_*` en logaritmo (`log1p`). Los umbrales de `integrate_derivates.py` ya están convertidos a esa escala.

En el clima, quita las horas duplicadas, completa la serie hora por hora, pasa a faltante los códigos de NASA POWER (`-999`), limita la humedad relativa y la nubosidad al rango de 0 a 100 e interpola los huecos de hasta 3 horas.

### 6. Integrar el clima

**Windows**

```powershell
py src/DP/integrar_clima.py
```

**macOS y Linux**

```bash
python3 src/DP/integrar_clima.py
```

**Resultado esperado:** se genera `Data/dataset_vacas_integrado.csv` a partir de los datasets limpios, con temperatura y THI de la sesión y horas con THI alto en las 24 horas previas.

### 7. Integrar las variables derivadas

**Windows**

```powershell
py src/DP/integrate_derivates.py
```

**macOS y Linux**

```bash
python3 src/DP/integrate_derivates.py
```

**Resultado esperado:** se actualiza `Data/dataset_vacas_integrado.csv` con las variables derivadas y etiquetas de eventos, conservando las filas y columnas existentes, incluidas las de clima. Los rechazos y las echadas sin ordeñar no reciben etiqueta de evento, y `fme_milk_production` no cuenta los cuartos marcados como inactivos (`cuarto_inactivo_*`). La terminal muestra estadísticas de las sesiones y eventos.

### 8. Construir, anonimizar e integrar el concentrado consumido


**Windows**

```powershell
py src/scripts/generate_concentrado.py
py src/scripts/anonymize_concentrado.py
py src/DP/integrar_concentrado.py
```

**macOS y Linux**

```bash
python3 src/scripts/generate_concentrado.py
python3 src/scripts/anonymize_concentrado.py
python3 src/DP/integrar_concentrado.py
```

**Resultado esperado:** se generan `Data/dataset_concentrado.csv` y `Data/dataset_concentrado_anonimo.csv`. El segundo usa los mismos hashes de vaca que ordeño, y el mapa no se modifica. Los archivos de `Data/` son locales y no se versionan en Git. `Data/dataset_vacas_integrado.csv` conserva todas sus filas y columnas previas, y añade las 4 variables definidas para el concentrado: `conc_sesion_previa`, `tope_sesion_previa`, `conc_d1` y `conc_media_7d`.

## Modelo LSTM

Se asume que se realizaron los pasos de la sección **¿Cómo obtener y preparar los datos?** y cuenta con `Data/dataset_vacas_integrado.csv`, incluidas las variables de clima y las variables derivadas. Ejecuta los siguientes comandos desde la raíz del repositorio.

### 1. Agregar las variables de historial

**Windows**

```powershell
py src/MODELING/LSTM/build_history_features.py
```

**macOS y Linux**

```bash
python3 src/MODELING/LSTM/build_history_features.py
```

**Resultado esperado:** se actualiza `Data/dataset_vacas_integrado.csv` con el calendario (`mes`, `hora`, `dia_semana`, `dias_en_leche`, `dias_en_hato`), los valores del ordeño previo (`{evento}_previo`, `mdi_previo`, `conductividad_max_previa`, `produccion_previa`, `cuarto_inactivo_previo`) y las tasas de la vaca y del hato.

### 2. Entrenar la LSTM

**Windows**

```powershell
mkdir src/MODELING/LSTM/modelos/lstm_base
py -u src/MODELING/LSTM/main.py > src/MODELING/LSTM/modelos/lstm_base/lstm_base.log 2>&1
```

**macOS y Linux**

```bash
mkdir -p src/MODELING/LSTM/modelos/lstm_base
python3 -u src/MODELING/LSTM/main.py >| src/MODELING/LSTM/modelos/lstm_base/lstm_base.log 2>&1
```

**Resultado esperado:** se generan en `src/MODELING/LSTM/modelos/lstm_base/` el modelo (`lstm.keras`), las métricas por evento (`metrics.json`), las curvas de pérdida (`history.json`) y el log del entrenamiento. La red es multitarea, con una salida para cada evento y otra para `alguna_anomalia` (lo contrario de `successful_milking_event`):

- Lee la secuencia de las 20 sesiones previas de la vaca. Cada paso tiene los valores de esa sesión: FME, `blood_diff_*`, `box_time_to_yield_ratio`, producción, MDI, conductividad máxima, cuarto inactivo, eventos y horas desde el ordeño anterior.
- Después de la capa LSTM se agrega el contexto de la sesión a predecir: intervalo, máquina, clima, sangre por máquina, calendario y carga del hato.
- Usa un corte cronológico 70/15/15 sin espacio entre periodos. Con val se detiene el entrenamiento (early stopping cuando el PR-AUC promedio de las 4 salidas en val no mejora en 5 épocas, hasta 50 épocas) y se elige el umbral para un recall ≥ 80 %. Las métricas se calculan en train, val, test y en test separado por vaca, con los mismos grupos que XGBoost y la regresión logística: conocidas (estuvieron en train, `test_vacas_conocidas`), solo en val (`test_vacas_solo_val`) y nuevas (su primer ordeño es en test, `test_vacas_nuevas`).

### 3. Generar las gráficas

**Windows**

```powershell
py -u src/MODELING/LSTM/plot_results.py
```

**macOS y Linux**

```bash
python3 -u src/MODELING/LSTM/plot_results.py
```

**Resultado esperado:** se generan en `src/MODELING/LSTM/modelos/lstm_base/graficas/` ocho gráficas a partir del modelo guardado, sin reentrenar: curvas de aprendizaje (pérdida, PR-AUC promedio y accuracy por época), ROC y precisión-recall, calibración, matrices de confusión, distribución de probabilidades y desempeño semanal en test; métricas de vacas conocidas, solo en val y nuevas; e importancia por permutación en val (caída del PR-AUC al revolver cada variable).

## Líneas base de la evaluación

Se asume que se realizaron los pasos de la sección **¿Cómo obtener y preparar los datos?** y cuenta con `Data/dataset_vacas_integrado.csv`. El script no necesita los modelos ni las variables de historial, y no modifica el CSV.

**Windows**

```powershell
py src/EVALUATION/lineas_base.py
```

**macOS y Linux**

```bash
python3 src/EVALUATION/lineas_base.py
```

**Resultado esperado:** se genera `src/EVALUATION/metrics/lineas_base.json` y la terminal muestra, para cada evento y para alguna anomalía, el F1, el recall y la precisión en test de tres reglas sin modelo:

- **Alertar siempre:** alerta en todos los ordeños; su F1 es 2p / (1 + p), con p la prevalencia.
- **Persistencia:** alerta si el ordeño anterior de la vaca, en el mismo bloque de historial, tuvo el evento.
- **Tasa de la vaca en 30 días:** alerta si la proporción de ordeños de la vaca con el evento en los 30 días previos supera el umbral fijado en val para un recall ≥ 80 %, con la misma regla que XGBoost.

Usa los mismos ordeños, etiquetas y corte cronológico 70/15/15 que los modelos (reutiliza `split_by_time` de XGBoost), así que sus resultados se comparan directo con los de test de cada modelo.

## Interfaz de predicción

Se asume que se realizaron los pasos de la sección **¿Cómo obtener y preparar los datos?** y `build_history_features.py` del XGBoost, y que está el modelo versionado en `src/MODELING/XGBoost/`. La interfaz no reentrena: carga el XGBoost de `versions.py`, el modelo final.

**Windows**

```powershell
py src/INTERFAZ/app.py
```

**macOS y Linux**

```bash
python3 src/INTERFAZ/app.py
```

En Mac, XGBoost necesita OpenMP: si falla al cargar `libxgboost.dylib`, instala `brew install libomp`.

**Resultado esperado:** la terminal muestra `[app] Lista en http://localhost:8000` (tarda unos segundos en cargar el dataset y el modelo). Al abrir esa dirección en el navegador:

- **Nueva sesión:** eliges una vaca y capturas lo que se sabe al entrar al robot (fecha y hora, máquina, número de ordeño, temperatura, humedad y horas con THI alto). El resto de las variables (FME, sangre, ordeño previo, tasas de la vaca y del hato y calendario) se calcula con el historial de la vaca, con las mismas funciones de `integrate_derivates.py` y `build_history_features.py`. La fecha puede ser a lo más el día siguiente al fin del dataset, porque las tasas del hato usan los 7 días anteriores; si pasaron más de 48 h desde el último ordeño o 45 días desde la última visita, la página lo avisa.
- **Del dataset:** eliges un ordeño que ya está en los datos y además ves si cada evento ocurrió.
- El XGBoost da la probabilidad calibrada de los 4 eventos, si pasa el umbral de alerta elegido en val y la variable que más aumenta el riesgo, con el mismo método que sus alertas de test (SHAP). Si ninguna variable sube el riesgo, no se muestra ninguna. Es una asociación, no una causa. La pestaña **Resultado**, que se abre al predecir, lo muestra en velocímetros y dice cuánto se alargaría el ordeño si ocurre cada evento (con `src/EVALUATION/metrics/duracion_por_evento.json`); **Resultado detallado** tiene la tabla y las tarjetas por evento.
- La pestaña **Desempeño del modelo** muestra F1, recall, precisión y Brier del XGBoost en test, y la línea base de persistencia de `src/EVALUATION/metrics/lineas_base.json`.

El [manual de uso de la interfaz](docs/DEPLOYMENT/Manual_interfaz.md) explica cada campo, cómo leer los resultados, los avisos y errores, y los problemas comunes.

## Preparación para reto de datos.

Para generar la muestra a partir de un dataset integrado existente, sigue los pasos 1 y 2. Primero debes completar la sección **¿Cómo obtener y preparar los datos?** y contar con `Data/dataset_vacas_integrado.csv`, incluidas las variables de clima y las variables derivadas.

Ejecuta los siguientes comandos desde la raíz del repositorio.

### 1. Obtener una muestra representativa de la población

Ejecuta `get_data_sample.py` para calcular el tamaño de muestra mediante la fórmula de Cochran con corrección por población finita y seleccionar registros al azar, estratificados proporcionalmente por `id_vaca`.

**Windows**

```powershell
py src/RETO_DATOS/get_data_sample.py
```

**macOS y Linux**

```bash
python3 src/RETO_DATOS/get_data_sample.py
```

**Resultado esperado:** se genera `src/RETO_DATOS/dataset_vacas_muestra.csv`, conservando todas las columnas del conjunto integrado. La terminal muestra el tamaño de la muestra y cuántos estratos están representados.

### 2. Separar los conjuntos de entrenamiento, validación y prueba

Ejecuta `k_fold.py` para separar primero un 20% de la muestra como conjunto de prueba (`test`). Este conjunto queda fijo y **no participa en k-fold**. Con el 80% restante se generan cinco folds: en cada uno, una parte se usa para validación (`val`) y las otras cuatro para entrenamiento (`train`).

**Windows**

```powershell
py src/RETO_DATOS/k_fold.py
```

**macOS y Linux**

```bash
python3 src/RETO_DATOS/k_fold.py
```

**Resultado esperado:** la terminal muestra el número de registros de `train`, `val` y `test` en cada fold, además de las primeras cinco filas de los conjuntos del primer fold.

Para obtener los conjuntos desde otro script, importa `get_k_folds` y selecciona un fold:

```python
from src.RETO_DATOS.k_fold import INPUT_DIR, get_k_folds

dataset, folds = get_k_folds(INPUT_DIR)
fold = folds[0]

train = dataset.iloc[fold["train"]]
val = dataset.iloc[fold["val"]]
test = dataset.iloc[fold["test"]]
```

> **Nota:** esta separación corresponde al reto de datos. Para el modelado, el [Reporte de diseño de pruebas](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.eaxe6c7fw3yo) define un corte cronológico 70/15/15: el 70 % más antiguo para entrenamiento, el 15 % siguiente para validación y el 15 % más reciente para prueba.

### 3. Medir el tiempo de ejecución del pipeline.

El ejecutor realiza diez vueltas completas de los once scripts de preparación, desde la generación de los datos hasta la muestra y la separación en folds. Antes de ejecutarlo, prepara los archivos originales de vacas y concentrado, configura `COWSKEY` e instala las dependencias como se indica en **¿Cómo obtener y preparar los datos?**.

**Windows**

```powershell
py src/RETO_DATOS/run_pipeline.py
```

**macOS y Linux**

```bash
python3 src/RETO_DATOS/run_pipeline.py
```

**Resultado esperado:** la consola muestra los datos del equipo, una estimación de la velocidad de descarga antes de cada vuelta y el tiempo de cada script. Al finalizar, presenta el promedio, la desviación estándar, el mínimo y el máximo de los tiempos por script y del pipeline completo. El clima se descarga nuevamente en cada vuelta.


## ¿Cómo acceder a los entregables?

### En GitHub

El repositorio concentra el reporte y los entregables de las fases del proyecto. 

La estructura del repositorio es:

- **[`docs/`](docs/)** — Documentación y entregables.
  - **[`BU/`](docs/BU/)** — Fase de *Entendimiento del negocio*.
    - [Reporte Business Understanding](docs/BU/Reportes%20de%20fase%20entendimiento%20del%20negocio.pdf).
    - [Reporte Business Understanding II](docs/BU/Reportes%20de%20fase%20entendimiento%20del%20negocio%20II.pdf): segunda iteración, cambia el criterio de éxito de minería de datos de accuracy a F1.
    - Memorándum de entendimiento.
    - One Page.
    - WBS y su archivo fuente.
  - **[`DU/`](docs/DU/)** — Fase de *Entendimiento de los datos*
    - [Reporte Data Understanding](docs/DU/Reportes%20de%20fase%20Data%20Understanding.pdf).
    - [Reporte Data Understanding II](docs/DU/Reportes%20de%20fase%20Data%20Understanding%20II.pdf): segunda iteración, con su changelog.
    - [Investigación de la fuente de clima](docs/DU/Investigacion_Clima.md).
  - **[`DP/`](docs/DP/)** — Fase de *Preparación de los datos*.
    - [Reporte Data Preparation](docs/DP/Reportes%20de%20fase%20Data%20Preparation.pdf).
    - [Reporte Data Preparation II](docs/DP/Reportes%20de%20fase%20Data%20Preparation%20II.pdf): segunda iteración, con su changelog.
  - **[`MODELING/`](docs/MODELING/)** — Fase de *Modelado*.
    - [Reporte Modeling](docs/MODELING/Reportes%20de%20fase%20Modeling.pdf): diseño de pruebas, regresión logística, XGBoost, LSTM y evaluación de los modelos.
    - [Reporte Modeling II](docs/MODELING/Modeling%20II.pdf): segunda iteración, con su changelog, diseño de pruebas y comparación de los modelos.
    - [Anexo I: Regresión logística](docs/MODELING/Anexo%20I%20-%20Regresi%C3%B3n%20logistica.pdf).
    - [Anexo II: XGBoost](docs/MODELING/Anexo%20II%20-%20XGBoost.pdf).
    - [Anexo III: LSTM](docs/MODELING/Anexo%20III%20-%20LSTM.pdf).
  - **[`EVALUATION/`](docs/EVALUATION/)** — Fase de *Evaluación*.
    - [Reporte Evaluation](docs/EVALUATION/Evaluation.pdf): evaluación de resultados frente a los criterios de éxito de negocio, revisión del proceso y decisión.
  - **[`DEPLOYMENT/`](docs/DEPLOYMENT/)** — Fase de *Despliegue*.
    - [Manual de uso de la interfaz](docs/DEPLOYMENT/Manual_interfaz.md).
  - **[`RETO_DATOS/`](docs/RETO_DATOS/)** — Preparación para el reto de datos.
    - [Reporte del reto de datos](docs/RETO_DATOS/Reporte-RetoDatos.pdf).
  - [`guide_audit_logs.md`](docs/guide_audit_logs.md) y [`data_audit_logs.csv`](docs/data_audit_logs.csv) — Guía y bitácora de auditoría de registros.
- **[`src/`](src/)** — Código fuente de apoyo.
  - **[`scripts/`](src/scripts/)** — Scripts de preparación de datos.
    - [`generate_data.py`](src/scripts/generate_data.py) — Procesa los archivos CSV y XLSX de cada vaca.
    - [`anonymize_data.py`](src/scripts/anonymize_data.py) — Anonimiza los identificadores de vaca y máquina.
    - [`descargar_clima.py`](src/scripts/descargar_clima.py) — Descarga el clima por hora de NASA POWER a `Data/clima/`.
    - [`generate_concentrado.py`](src/scripts/generate_concentrado.py) y [`anonymize_concentrado.py`](src/scripts/anonymize_concentrado.py) — Consolidan y anonimizan los XLSX de concentrado.
  - **[`DU/`](src/DU/)** — Notebook de exploración para la fase *Data Understanding*.
    - [`final_data_understanding.ipynb`](src/DU/final_data_understanding.ipynb) — Reúne los análisis exploratorios del equipo sobre el dataset anonimizado, el clima de NASA POWER y el concentrado.
  - **[`DP/`](src/DP/)** — Scripts de la fase *Data Preparation*.
    - [`clean_data.py`](src/DP/clean_data.py) — Limpia el dataset anonimizado de vacas y el clima de NASA POWER, y genera `Data/dataset_vacas_limpio.csv` y `Data/clima/clima_nasa_power_limpio.csv`.
    - [`integrar_clima.py`](src/DP/integrar_clima.py) — Agrega a las sesiones limpias los atributos de clima de NASA POWER (temperatura, THI y horas con THI alto) y crea `Data/dataset_vacas_integrado.csv`.
    - [`integrate_derivates.py`](src/DP/integrate_derivates.py) — Agrega diez atributos derivados del historial de cada vaca y cuatro etiquetas de evento a `Data/dataset_vacas_integrado.csv`.
    - [`integrar_concentrado.py`](src/DP/integrar_concentrado.py) — Añade cuatro atributos históricos de concentrado a `Data/dataset_vacas_integrado.csv`.
  - **[`MODELING/`](src/MODELING/)** — Scripts de la fase *Modeling*. Los tres modelos usan el corte cronológico 70/15/15 y tienen una salida por evento de ordeño y otra para alguna anomalía.
    - **[`logistic/`](src/MODELING/logistic/)** — Regresión logística (versiones v1–v4; `VERSION` en `main.py` elige cuál se guarda).
      - [`main.py`](src/MODELING/logistic/main.py) — Entrena cuatro regresiones con train y las evalúa con validación y test. Usa `dataset.py`, `evaluation.py` y `logistic_model.py`.
      - [`dataset.py`](src/MODELING/logistic/dataset.py) — Carga los datos y el historial disponible antes de cada sesión, sin modificar el CSV original.
      - [`logistic_model.py`](src/MODELING/logistic/logistic_model.py) — Rellena con cero y bandera, escala con train y ajusta la regresión logística.
      - [`evaluation.py`](src/MODELING/logistic/evaluation.py) — Selección con validación y evaluación de probabilidades y decisiones binarias.
      - [`plot_results.py`](src/MODELING/logistic/plot_results.py) — Gráficas de evaluación y del avance por iteraciones en [`img/`](src/MODELING/logistic/img/).
      - [`models/`](src/MODELING/logistic/models/) y [`metrics/`](src/MODELING/logistic/metrics/) — Modelos y métricas de cada versión.
    - **[`XGBoost/`](src/MODELING/XGBoost/)** — XGBoost calibrado por evento (versiones v1–v5; la final es v5).
      - [`build_history_features.py`](src/MODELING/XGBoost/build_history_features.py) — Agrega las 21 variables de historial que usa el XGBoost a `Data/dataset_vacas_integrado.csv`.
      - [`versions.py`](src/MODELING/XGBoost/versions.py) — Hiperparámetros de cada versión; `VERSION` elige cuál corren `main.py` y `plot_results.py`.
      - [`main.py`](src/MODELING/XGBoost/main.py) — Entrena, calibra y evalúa la versión elegida y guarda los modelos y las métricas en [`models/<versión>/`](src/MODELING/XGBoost/models/). Usa `dataset.py`, `evaluation.py` y `xgboost_model.py`.
      - [`random_search.py`](src/MODELING/XGBoost/random_search.py) — Búsqueda al azar de hiperparámetros, medida solo en train y val; la mejor combinación es v5 (`models/v5/busqueda.csv`).
      - [`evaluation.py`](src/MODELING/XGBoost/evaluation.py) — Umbral de recall, métricas por conjunto y los 3 factores SHAP de cada alerta.
      - [`plot_results.py`](src/MODELING/XGBoost/plot_results.py) — Genera con `plots.py` las gráficas de la versión elegida en [`img/<versión>/`](src/MODELING/XGBoost/img/), sin reentrenar.
    - **[`LSTM/`](src/MODELING/LSTM/)** — LSTM multitarea (versiones v1–v5; la final es v5).
      - [`build_history_features.py`](src/MODELING/LSTM/build_history_features.py) — Agrega las variables de calendario, ordeño previo y tasas de la vaca y del hato a `Data/dataset_vacas_integrado.csv`.
      - [`dataset.py`](src/MODELING/LSTM/dataset.py) — Carga los ordeños, hace el corte cronológico y arma las secuencias y el contexto de la LSTM.
      - [`main.py`](src/MODELING/LSTM/main.py) — Entrena, calibra y evalúa la LSTM y guarda el modelo y las métricas en [`modelos/<versión>/`](src/MODELING/LSTM/modelos/) y [`metrics/<versión>/`](src/MODELING/LSTM/metrics/).
      - [`evaluation.py`](src/MODELING/LSTM/evaluation.py) — Umbral, métricas, calibración y tablas CSV.
      - [`explain.py`](src/MODELING/LSTM/explain.py) — Los 3 factores de cada alerta con Integrated Gradients.
      - [`plot_results.py`](src/MODELING/LSTM/plot_results.py) — Genera las gráficas de evaluación de la LSTM en [`img/<versión>/`](src/MODELING/LSTM/img/).
  - **[`EVALUATION/`](src/EVALUATION/)** — Scripts de la fase *Evaluation*.
    - [`lineas_base.py`](src/EVALUATION/lineas_base.py) — Mide en test tres líneas base sin modelo (alertar siempre, persistencia y tasa de la vaca en 30 días) con el mismo corte 70/15/15 de los modelos, y las guarda en [`metrics/lineas_base.json`](src/EVALUATION/metrics/lineas_base.json).
  - **[`INTERFAZ/`](src/INTERFAZ/)** — Interfaz web para probar el modelo final (XGBoost).
    - [`app.py`](src/INTERFAZ/app.py) — Servidor Flask: sirve la página y la API de predicción.
    - [`nueva_sesion.py`](src/INTERFAZ/nueva_sesion.py) — Arma las variables de una sesión capturada con el historial de la vaca.
    - [`modelos.py`](src/INTERFAZ/modelos.py) — Carga el XGBoost y da la probabilidad y la variable más importante de cada evento.
    - [`traduccion.py`](src/INTERFAZ/traduccion.py) — Nombres reales de vacas y máquinas, y el glosario de variables en lenguaje no técnico.
    - [`tiempos.py`](src/INTERFAZ/tiempos.py) — Cuánto se alarga el ordeño si ocurre cada evento (métricas de `src/EVALUATION/duracion_por_evento.py`) frente a lo normal de la vaca.
    - [`static/`](src/INTERFAZ/static/) — La página (`index.html`), sus estilos (`styles.css`), su código (`script.js`) y los íconos (`iconos.svg`, `favicon.svg`).
  - **[`RETO_DATOS/`](src/RETO_DATOS/)** — Código y datos del reto de datos.
    - [`get_data_sample.py`](src/RETO_DATOS/get_data_sample.py) — Obtiene una muestra estratificada por vaca con la fórmula de Cochran.
    - [`dataset_vacas_muestra.csv`](src/RETO_DATOS/dataset_vacas_muestra.csv) — Muestra generada por el script anterior.
    - [`k_fold.py`](src/RETO_DATOS/k_fold.py) — Separa la muestra en un conjunto de prueba fijo y cinco folds de entrenamiento y validación.
    - [`run_pipeline.py`](src/RETO_DATOS/run_pipeline.py) — Ejecuta diez veces la preparación completa y resume sus tiempos.

Los entregables de las siguientes fases se incorporarán a esta estructura conforme se desarrollen.


### En Google Drive

El reporte está en un solo documento con una pestaña por fase. Los números de página son los del índice de cada pestaña.

#### [Reporte: Entendimiento del negocio](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.0)

<pre>
<strong>Adaptaciones de la metodología CRISP-DM.................1</strong>
<strong>Antecedentes............................................1</strong>
<strong>Objetivo de negocio.....................................2</strong>
    Criterios de éxito de negocio.......................3
<strong>Objetivo de minería de datos............................3</strong>
    Criterios de éxito de minería de datos..............3
<strong>Requerimientos del proyecto.............................3</strong>
<strong><a href="https://docs.google.com/spreadsheets/d/1cll5V8mhmh2yNx-Pf1LDWkmkLIdHjaAfX-hKlmNiJMg/edit?usp=sharing">Plan de valor ganado</a>....................................4</strong>
<strong>Supuestos del proyecto..................................4</strong>
<strong>Limitaciones del proyecto...............................4</strong>
<strong><a href="https://docs.google.com/spreadsheets/d/1WDTALeblrlLVCPUdWbw-YFCtojI6zTDLFODQ9f9Gr6Y/edit?usp=sharing">Inventario de recursos</a>..................................5</strong>
<strong><a href="https://docs.google.com/document/d/1eC2AxkIQMhPYsphopbgM1sPM5fT8XvkHgX1WiRlCP2M/edit?usp=sharing">Análisis de costo y beneficio</a>...........................5</strong>
<strong><a href="https://docs.google.com/spreadsheets/d/15yn73QzDFNr9cexEtcF1Ktdak3ik_enyfIcaNqOjK60/edit?usp=sharing">Matriz de riesgos</a>.......................................5</strong>
<strong>Glosario de terminología................................5</strong>
<strong>Referencias.............................................7</strong>
</pre>

#### [Reporte: Entendimiento del negocio II](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.7ytazc1p5l16)

<pre>
<strong>Adaptaciones de la metodología CRISP-DM.................1</strong>
<strong>Antecedentes............................................1</strong>
<strong>Objetivo de negocio.....................................2</strong>
    Criterios de éxito de negocio.......................3
<strong>Objetivo de minería de datos............................3</strong>
    Criterios de éxito de minería de datos..............3
<strong>Requerimientos del proyecto.............................3</strong>
<strong><a href="https://docs.google.com/spreadsheets/d/1cll5V8mhmh2yNx-Pf1LDWkmkLIdHjaAfX-hKlmNiJMg/edit?usp=sharing">Plan de valor ganado</a>....................................4</strong>
<strong>Supuestos del proyecto..................................4</strong>
<strong>Limitaciones del proyecto...............................4</strong>
<strong><a href="https://docs.google.com/spreadsheets/d/1WDTALeblrlLVCPUdWbw-YFCtojI6zTDLFODQ9f9Gr6Y/edit?usp=sharing">Inventario de recursos</a>..................................5</strong>
<strong><a href="https://docs.google.com/document/d/1eC2AxkIQMhPYsphopbgM1sPM5fT8XvkHgX1WiRlCP2M/edit?usp=sharing">Análisis de costo y beneficio</a>...........................5</strong>
<strong><a href="https://docs.google.com/spreadsheets/d/15yn73QzDFNr9cexEtcF1Ktdak3ik_enyfIcaNqOjK60/edit?usp=sharing">Matriz de riesgos</a>.......................................5</strong>
<strong>Glosario de terminología................................5</strong>
<strong>Referencias.............................................7</strong>
</pre>

#### [Reporte: Entendimiento de los datos](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.vtkudwv13p69)

<pre>
<strong>Resumen..................................................1</strong>
<strong>1. Adaptaciones de la metodología CRISP-DM...............2</strong>
<strong>2. Recolección de datos iniciales........................2</strong>
    2.1 Ubicación y obtención............................2
    2.2 Problemas encontrados y soluciones...............3
<strong>3. Descripción de los datos..............................3</strong>
    3.1 Fuentes..........................................3
    3.2 Formato y tipos de datos.........................4
    3.3 Descripción de columnas del dataset..............5
    3.4 Características del dataset......................6
    3.5 Clima............................................7
<strong>4. Satisfacción de los requerimientos....................7</strong>
<strong>5. Exploración de los datos..............................9</strong>
    5.1 Visualizar y consultar los datos.................10
    5.2 Patrones y distribuciones........................11
    5.3 Problema de cola.................................14
    5.4 Clima integrado..................................16
    5.5 Hipótesis iniciales..............................18
<strong>6. Calidad de los datos..................................18</strong>
    6.1 Análisis de la calidad...........................18
    6.2 Datos atípicos...................................20
    6.3 Métodos de limpieza recomendados.................21
<strong>7. Implicaciones para el modelado........................22</strong>
    Anexo: calidad por columna de los historiales CSV....22
<strong>8. Glosario..............................................24</strong>
</pre>

#### [Reporte: Entendimiento de los datos II](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.rhwdgmtp2831)

<pre>
<strong>Changelog respecto a la iteración anterior...............1</strong>
<strong>Resumen..................................................2</strong>
<strong>1. Adaptaciones de la metodología CRISP-DM...............2</strong>
<strong>2. Recolección de datos iniciales........................2</strong>
    2.1 Ubicación y obtención............................2
    2.2 Problemas encontrados y soluciones...............3
<strong>3. Descripción de los datos..............................4</strong>
    3.1 Fuentes..........................................4
    3.2 Formato y tipos de datos.........................5
    3.3 Descripción de columnas del dataset original.....6
        Identificación y resumen (main_*)............6
        Estado (estado_*)............................6
        Mediciones por cuarto (DI, DD, TI, TD).......7
        Contexto (misc_*)............................7
        Contadores de monitorización.................8
    3.4 Características del dataset......................8
    3.5 Clima............................................10
<strong>4. Satisfacción de los requerimientos....................10</strong>
<strong>5. Exploración de los datos..............................12</strong>
    5.1 Visualizar y consultar los datos.................13
    5.2 Patrones y distribuciones........................14
    5.3 Problema de cola.................................17
    5.4 Clima integrado..................................19
    5.5 Hipótesis iniciales..............................21
<strong>6. Calidad de los datos..................................21</strong>
    6.1 Análisis de la calidad...........................21
    6.2 Datos atípicos...................................23
    6.3 Métodos de limpieza recomendados.................24
<strong>7. Implicaciones para el modelado........................25</strong>
    Anexo: calidad por columna de los historiales CSV....25
<strong>8. Glosario..............................................27</strong>
</pre>

#### [Reporte: Preparación de los datos](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.dll5kcgopm0k)

<pre>
<strong>Adaptaciones de la metodología CRISP-DM.................................1</strong>
<strong>Reporte de limpieza de datos............................................2</strong>
<strong>Reporte de construcción e integración de datos..........................4</strong>
    Integración del clima con las sesiones de ordeño....................4
    Construcción de los atributos de temperatura........................4
    Construcción de los atributos derivados.............................5
    Construcción de los atributos de concentrado........................9
<strong>Reporte de formato de datos.............................................12</strong>
    Investigar posibles modelos en base a los datos.....................12
<strong>Reporte de selección del modelo.........................................14</strong>
    Técnica seleccionada de los modelos.................................14
    Supuestos de los modelos............................................15
    Técnica de modelado y comparación justa entre modelos...............17
    Ajustes a la preparación de los datos requeridos por los modelos....18
<strong>Reporte de selección de datos...........................................19</strong>
<strong>Glosario de terminología................................................26</strong>
<strong>Fuentes bibliográficas..................................................28</strong>
</pre>

#### [Reporte: Preparación de los datos II](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.d0jw2eth3804)

<pre>
<strong>Adaptaciones de la metodología CRISP-DM.................................1</strong>
<strong>Reporte de limpieza de datos............................................2</strong>
<strong>Reporte de construcción e integración de datos..........................4</strong>
    Integración del clima con las sesiones de ordeño....................4
    Construcción de los atributos de temperatura........................4
    Construcción de los atributos derivados.............................5
    Construcción de los atributos de concentrado........................9
<strong>Reporte de formato de datos.............................................12</strong>
    Investigar posibles modelos en base a los datos.....................12
<strong>Reporte de selección del modelo.........................................14</strong>
    Técnica seleccionada de los modelos.................................14
    Supuestos de los modelos............................................15
    Técnica de modelado y comparación justa entre modelos...............17
    Ajustes a la preparación de los datos requeridos por los modelos....18
<strong>Reporte de selección de datos...........................................19</strong>
<strong>Glosario de terminología................................................26</strong>
<strong>Fuentes bibliográficas..................................................28</strong>
</pre>

#### [Reporte: Modelado de los datos](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.7tfnlrrsx4yq)

<pre>
<strong>Adaptaciones de la metodología CRISP-DM.......................2</strong>
<strong>Diseño de pruebas.............................................2</strong>
    Partición de los datos....................................3
    Plan de entrenamiento y ajuste............................4
    Evaluación y comparación de los modelos...................5
    Regla de elección del modelo final........................6
<strong>Construcción del modelo de regresión logística................7</strong>
    Comparativa de las versiones de los modelos...............7
    Hiperparámetros usados....................................8
    Preparación de los datos..................................10
    Evaluación del desempeño del modelo.......................11
    Interpretación de los resultados..........................21
    Dificultades identificadas................................22
    Vacas nuevas en test......................................23
<strong>Construcción del modelo XGBoost...............................26</strong>
    Comparativa de las versiones de los modelos...............26
    Hiperparámetros usados....................................27
    Evaluación del desempeño del modelo.......................28
    Interpretación de los resultados..........................35
    Dificultades identificadas................................35
    Vacas nuevas en test......................................36
    Interpretación con SHAP...................................38
<strong>Construcción del modelo LSTM..................................39</strong>
    Comparativa de las versiones de los modelos...............39
    Hiperparámetros usados....................................41
    Preparación de los datos..................................43
    Evaluación del desempeño del modelo.......................44
    Criterios del BU en test..................................47
    Gráficos..................................................48
    Interpretación de los resultados..........................52
    Dificultades identificadas................................52
    Vacas nuevas en test......................................53
    Conclusión................................................54
<strong>Evaluación de modelos.........................................55</strong>
    Comparación de resultados.................................55
    Cumplimiento de criterios establecidos....................56
    Diferencias en el comportamiento..........................57
    Dificultades en los modelos...............................58
    Elección de modelo........................................58
</pre>

#### [Reporte: Modelado de los datos II](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.eaxe6c7fw3yo)

<pre>
<strong>Changelog respecto a la iteración anterior de Modeling..........2</strong>
<strong>Adaptaciones de la metodología CRISP-DM.........................2</strong>
<strong>Diseño de pruebas...............................................3</strong>
    Partición de los datos......................................4
    Plan de entrenamiento y ajuste..............................5
    Evaluación y comparación de los modelos.....................6
    Regla de elección de modelos finales........................6
<strong>Construcción del modelo de regresión logística..................7</strong>
    Qué hace el modelo..........................................7
    Cómo se entrenó y se eligió la versión......................8
    Resultado en test...........................................9
    Importancia de las variables y 3 factores de cada alerta...11
    Dificultades identificadas.................................12
    Vacas nuevas en test.......................................12
    Conclusión.................................................12
<strong>Construcción del modelo XGBoost................................13</strong>
    Qué hace el modelo.........................................13
    Cómo se entrenó y se eligió la versión.....................14
    Resultado en test..........................................16
    Importancia de las variables y 3 factores de cada alerta...17
    Dificultades identificadas.................................19
    Vacas nuevas en test.......................................20
    Conclusión.................................................20
<strong>Construcción del modelo LSTM...................................21</strong>
    Qué hace el modelo.........................................21
    Cómo se entrenó y se eligió la versión.....................22
    Resultado en test..........................................23
    Importancia de las variables y 3 factores de cada alerta...24
    Criterios del BU en test...................................25
    Dificultades identificadas.................................26
    Vacas nuevas en test.......................................27
    Conclusión.................................................27
<strong>Evaluación de modelos..........................................27</strong>
    Comparación de resultados..................................27
    Cumplimiento de criterios establecidos.....................28
    Diferencias en el comportamiento...........................29
    Dificultades en los modelos................................29
<strong>Referencias....................................................30</strong>
</pre>

#### [Anexo 1: Regresión logística](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.o0oj2yrcn78s)

<pre>
<strong>Comparativa de las versiones de los modelos.....................1</strong>
<strong>Hiperparámetros usados..........................................3</strong>
<strong>Preparación de los datos........................................4</strong>
<strong>Evaluación del desempeño del modelo.............................6</strong>
    Falla mecánica..............................................6
    Anomalía de salud...........................................7
    Interrupción por conducta...................................8
    Alguna anomalía.............................................8
<strong>Criterios del BU en test........................................9</strong>
<strong>Gráficos.......................................................10</strong>
    Curvas de aprendizaje: Pérdida logarítmica.................10
    Matrices de confusión (en el conjunto de test).............12
    ROC y PR (en el conjunto de test)..........................13
    Calibración (en el conjunto de test).......................13
<strong>Interpretación de los resultados...............................15</strong>
<strong>Dificultades identificadas.....................................16</strong>
<strong>Vacas nuevas en test...........................................17</strong>
    Desempeño por grupo de vaca................................18
    Calidad de las probabilidades por grupo....................19
<strong>Conclusión.....................................................19</strong>
</pre>

#### [Anexo 2: XGBoost](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.xpypjci3s25w)

<pre>
<strong>Comparativa de las versiones del modelo.........................1</strong>
<strong>Hiperparámetros usados..........................................2</strong>
<strong>Preparación de los datos........................................3</strong>
<strong>Evaluación del desempeño del modelo.............................3</strong>
    Falla mecánica (umbral de 0.120)............................3
    Anomalía de salud (umbral de 0.208).........................4
    Interrupción por conducta (umbral de 0.128).................4
    Alguna anomalía (umbral de 0.338)...........................4
<strong>Criterios del BU en test........................................5</strong>
<strong>Gráficos........................................................5</strong>
    Curvas de aprendizaje.......................................5
    Calibración en test.........................................6
    Distribución de probabilidades en test......................6
    Importancia por gain........................................7
    Importancia SHAP............................................7
    Desempeño semanal en test...................................8
<strong>Vacas nuevas en test............................................8</strong>
</pre>

#### [Anexo 3: LSTM](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.73mzsvi35mgs)

<pre>
<strong>Comparativa de las versiones del modelo.........................1</strong>
<strong>Hiperparámetros usados..........................................2</strong>
<strong>Preparación de los datos........................................4</strong>
<strong>Evaluación del desempeño del modelo.............................5</strong>
<strong>Criterios del BU en test........................................6</strong>
<strong>3 factores por alerta...........................................7</strong>
<strong>Gráficos........................................................7</strong>
    Curvas de aprendizaje.......................................7
    Calibración en test.........................................8
    Importancia con Integrated Gradients........................9
<strong>Vacas nuevas en test............................................9</strong>
</pre>

#### [Reporte: Evaluación](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.wxkcbo5t63ty)

<pre>
<strong>Reporte: Evaluación..........................................................1</strong>
    Por PINU.................................................................1
<strong>Reporte de evaluación de resultados..........................................2</strong>
    Resultados de minería de datos...........................................2
        Interpretar en términos de la aplicación.............................2
        Impactos del objetivo de minería.....................................3
        Comparar con la base de conocimiento.................................4
    Evaluar con los criterios de éxito de negocio............................5
        Comparar la evaluación con su interpretación.........................5
        Clasificar los resultados según los criterios de éxito de negocio....6
        Impactos contra el objetivo inicial..................................7
        Declaración final....................................................7
    Modelos aprobados........................................................8
<strong>Revisar el Proceso...........................................................8</strong>
    Business Understanding...................................................8
    Data Understanding.......................................................8
    Data Preparation.........................................................9
    Modeling.................................................................9
    Pasos engañosos.........................................................10
    Resultados frente a los criterios de éxito..............................10
<strong>Lista de posibles acciones..................................................10</strong>
<strong>Decisión....................................................................11</strong>
    Nuevos objetivos o proyectos............................................12
<strong>Conclusiones para futuros proyectos.........................................12</strong>
</pre>
