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

### 5. Integrar el clima

**Windows**

```powershell
py src/DP/integrar_clima.py
```

**macOS y Linux**

```bash
python3 src/DP/integrar_clima.py
```

**Resultado esperado:** se genera `Data/dataset_vacas_integrado.csv` a partir de los datos anonimizados, con temperatura y THI de la sesión y horas con THI alto en las 24 horas previas.

### 6. Integrar las variables derivadas

**Windows**

```powershell
py src/DP/integrate_derivates.py
```

**macOS y Linux**

```bash
python3 src/DP/integrate_derivates.py
```

**Resultado esperado:** se actualiza `Data/dataset_vacas_integrado.csv` con las variables derivadas y etiquetas de eventos, conservando las filas y columnas existentes, incluidas las de clima. La terminal muestra estadísticas de las sesiones y eventos.

### 7. Construir, anonimizar e integrar el concentrado consumido


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
    - Guía y archivo CSV de auditoría de registros.
    - [Investigación de la fuente de clima](docs/DU/Investigacion_Clima.md).
- **[`src/`](src/)** — Código fuente de apoyo.
  - **[`scripts/`](src/scripts/)** — Scripts de preparación de datos.
    - [`generate_data.py`](src/scripts/generate_data.py) — Procesa los archivos CSV y XLSX de cada vaca.
    - [`anonymize_data.py`](src/scripts/anonymize_data.py) — Anonimiza los identificadores de vaca y máquina.
    - [`descargar_clima.py`](src/scripts/descargar_clima.py) — Descarga el clima por hora de NASA POWER a `Data/clima/`.
    - [`generate_concentrado.py`](src/scripts/generate_concentrado.py) y [`anonymize_concentrado.py`](src/scripts/anonymize_concentrado.py) — Consolidan y anonimizan los XLSX de concentrado.
  - **[`DU/`](src/DU/)** — Notebook de exploración para la fase *Data Understanding*.
    - [`final_data_understanding.ipynb`](src/DU/final_data_understanding.ipynb) — Reúne los análisis exploratorios del equipo sobre el dataset anonimizado, el clima de NASA POWER y el concentrado.
  - **[`DP/`](src/DP/)** — Scripts de la fase *Data Preparation*.
    - [`integrar_clima.py`](src/DP/integrar_clima.py) — Agrega a las sesiones los atributos de clima de NASA POWER (temperatura, THI y horas con THI alto) en `Data/dataset_vacas_integrado.csv`.
    - [`integrate_derivates.py`](src/DP/integrate_derivates.py) — Agrega diez atributos derivados del historial de cada vaca y cuatro etiquetas de evento a `Data/dataset_vacas_integrado.csv`.
    - [`integrar_concentrado.py`](src/DP/integrar_concentrado.py) — Añade cuatro atributos históricos de concentrado a `Data/dataset_vacas_integrado.csv`.

Los entregables de las siguientes fases se incorporarán a esta estructura conforme se desarrollen.


### En Google Drive

####  [Reporte: Entendimiento de los datos](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.0)

<pre>
<strong>Antecedentes............................................1</strong>
<strong>Objetivo de negocio.....................................2</strong>
    Criterios de éxito de negocio.......................2
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
<strong>Adaptaciones de la metodología CRISP-DM.................7</strong>
</pre>

#### [Reporte: Entendimiento de los datos](https://docs.google.com/document/d/1z871uI4bhOET6Eb1gTy_JprbtWaXhmogy0pr0fnJYk4/edit?tab=t.vtkudwv13p69)

<pre>
<strong>Reporte de recolección de datos iniciales...............1</strong>
    Ubicación y obtención de los datos..................1
    Problemas encontrados al obtener los datos..........2
    Soluciones a los problemas encontrados..............3
<strong>Reporte de descripción de los datos.....................4</strong>
    Formato y tipo de los datos.........................4
        Tipos de variable...............................5
    Particularidades de formato que requieren limpieza..5
    Características de los datos........................6
        Volumen y cobertura.............................6
        Distribución por vaca...........................6
        Clima (descarga exploratoria, aún sin integrar).6
<strong>Satisfacción de los requerimientos de los datos.........7</strong>
<strong>Conclusión sobre la calidad de los datos................8</strong>
<strong>Adaptaciones de la metodología CRISP-DM.................8</strong>
</pre>
