# Vaca_AI

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
- **[`src/`](src/)** — Código fuente de apoyo.
  - **[`DU/`](src/DU/)** — Scripts de preparación de datos para la fase *Data Understanding*.
    - [`generate_data.py`](src/DU/generate_data.py) — Procesa los archivos CSV de cada vaca.
    - [`anonymize_data.py`](src/DU/anonymize_data.py) — Anonimiza los identificadores de vaca y máquina.
    - [`descargar_clima.py`](src/DU/descargar_clima.py) — Descarga el clima por hora de NASA POWER a `Data/clima/`.
  - **[`DP/`](src/DP/)** — Scripts de la fase *Data Preparation*.
    - [`integrar_clima.py`](src/DP/integrar_clima.py) — Limpia las sesiones y les une el clima de NASA POWER en `Data/dataset_vacas_clima.csv`.

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
