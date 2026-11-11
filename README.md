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

## Preparación para reto de datos.

Se asume que se realizaron los pasos de la sección **¿Cómo obtener y preparar los datos?** y cuenta con `Data/dataset_vacas_integrado.csv`, incluidas las variables de clima y las variables derivadas. 

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
py src/DP/k_fold.py
```

**macOS y Linux**

```bash
python3 src/DP/k_fold.py
```

**Resultado esperado:** la terminal muestra el número de registros de `train`, `val` y `test` en cada fold, además de las primeras cinco filas de los conjuntos del primer fold.

Para obtener los conjuntos desde otro script, importa `get_k_folds` y selecciona un fold:

```python
from src.DP.k_fold import INPUT_DIR, get_k_folds

dataset, folds = get_k_folds(INPUT_DIR)
fold = folds[0]

train = dataset.iloc[fold["train"]]
val = dataset.iloc[fold["val"]]
test = dataset.iloc[fold["test"]]
```

> **Nota:** esta separación al azar corresponde al reto de datos. Para el modelado, el [Reporte de diseño de pruebas](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.7tfnlrrsx4yq) define un corte cronológico 70/15/15: el 70 % más antiguo para entrenamiento, el 15 % siguiente para validación y el 15 % más reciente para prueba.


## ¿Cómo acceder a los entregables?

### En GitHub

El repositorio concentra el reporte y los entregables de las fases del proyecto. 

La estructura del repositorio es:

- **[`docs/`](docs/)** — Documentación y entregables.
  - **[`BU/`](docs/BU/)** — Fase de *Entendimiento del negocio*.
    - Reporte Business Understanding.
    - Memorándum de entendimiento.
    - One Page.
    - WBS y su archivo fuente.
  - **[`DU/`](docs/DU/)** — Fase de *Entendimiento de los datos*
    - [Investigación de la fuente de clima](docs/DU/Investigacion_Clima.md).
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
    - [`k_fold.py`](src/DP/k_fold.py) — Separa la muestra en un conjunto de prueba fijo y cinco folds de entrenamiento y validación.
  - **[`MODELING/XGBoost/`](src/MODELING/XGBoost/)** — XGBoost base por evento de ordeño.
    - [`build_history_features.py`](src/MODELING/XGBoost/build_history_features.py) — Agrega las variables de historial de la tabla de selección de datos a `Data/dataset_vacas_integrado.csv`.
    - [`main.py`](src/MODELING/XGBoost/main.py) — Entrena y evalúa un clasificador por evento y uno para alguna anomalía con el corte cronológico 70/15/15. Usa `dataset.py`, `evaluation.py` y `xgboost_model.py`.
    - [`plot_results.py`](src/MODELING/XGBoost/plot_results.py) — Genera las gráficas del modelo con `plots.py` en [`img/`](src/MODELING/XGBoost/img/).
  - **[`RETO_DATOS/`](src/RETO_DATOS/)** — Código y datos del reto de datos.
    - [`get_data_sample.py`](src/RETO_DATOS/get_data_sample.py) — Obtiene una muestra estratificada por vaca con la fórmula de Cochran.
    - [`dataset_vacas_muestra.csv`](src/RETO_DATOS/dataset_vacas_muestra.csv) — Muestra generada por el script anterior.

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

#### [Reporte: Modelado de los datos](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.7tfnlrrsx4yq)

<pre>
<strong>Adaptaciones de la metodología CRISP-DM.......................1</strong>
<strong>Construcción del modelo de regresión logística (pendiente)....1</strong>
<strong>Construcción del modelo XGBoost...............................1</strong>
    Hiperparámetros usados....................................1
    Evaluación del desempeño del modelo.......................2
        Falla mecánica........................................2
        Anomalía de salud.....................................3
        Interrupción por conducta.............................3
        Criterios del BU en test..............................4
        Gráficos..............................................5
    Interpretación de los resultados..........................8
    Dificultades identificadas................................9
<strong>Construcción del modelo LSTM..................................9</strong>
<strong>Reporte de diseño de pruebas..................................10</strong>
</pre>
