# **Reporte: Entendimiento de los datos**
**Por PINU**

Fase *Data Understanding* (CRISP-DM). Resume la descripción, la exploración y la calidad de los datos del sistema de ordeño del CAETEC. Todas las cifras salen de [`final_data_understanding.ipynb`](../../src/DU/final_data_understanding.ipynb), donde está el detalle completo. Los resultados son exploratorios: describen asociaciones y no demuestran causas.

# **Resumen**

- **Los datos sirven para el objetivo, con tratamientos claros.** Hay 280,697 ordeños de 207 vacas en tres robots (13/07/2022 a 28/01/2026), con identificador de vaca en el 100 % de las filas y clima de NASA POWER para el 100 % de las sesiones. Necesitan limpieza (dos formatos de fecha, sesiones repetidas entre fuentes, ceros de conductividad que son faltantes y columnas vacías), pero nada impide modelar.
- **Los eventos son el 12 al 16 % de los ordeños.** Un modelo que siempre responda «sin evento» ya tiene 84 a 88 % de accuracy por evento, así que el criterio de 80 % no basta por sí solo.
- **El clima casi no se relaciona con el ordeño** (|r| ≤ 0.03 con los eventos). Pesan más el historial de la vaca (un evento previo multiplica de 3.3 a 4.5 veces la probabilidad del siguiente), el intervalo desde el ordeño anterior y el estado de los cuartos.
- **Hay señales de problema de cola, pero no se mide directamente:** la ocupación de las máquinas pasó de 13 % a 71 % y los relevos pegados de 21 % a 80 % entre 2022 y 2026, con las mismas tres máquinas.
- **Pendientes con el socio formador:** definir las clases de evento, las pausas de las 4, 12 y 20 h, el sensor de conductividad del cuarto trasero derecho y la unidad del concentrado.

# **1. Reporte de descripción de datos**

## 1.1 Datos y fuentes

| Fuente | Detalle |
| :---- | :---- |
| Ordeño | `Data/dataset_vacas_anonimo.csv`: reportes csv (marzo a julio de 2025) y históricos xlsx (2022 a 2026). 286,217 filas y 44 columnas; cada fila es una visita de una vaca a una máquina. |
| Clima | NASA POWER, serie horaria en hora local, 13 variables (ver [Investigacion_Clima.md](Investigacion_Clima.md)). |
| Concentrado | `Data/dataset_concentrado_anonimo.csv` (`vacas/4.0`): 129,532 registros de 202 vacas, del 28/01/2025 al 28/01/2026. |

Tras la limpieza quedan 280,915 filas: 280,697 ordeños de las tres máquinas (base del análisis), 216 visitas rechazadas o de «echada sin ordeñar» (solo en 2025) y 2 sesiones de un cuarto valor de máquina.

## 1.2 Formato y tipos de datos

El archivo original tiene 30 columnas numéricas (`float64`) y 14 de texto. Pandas asigna un tipo distinto al real en varias:

| Columna | Problema | Tratamiento |
| :---- | :---- | :---- |
| `main_hora_de_inicio` | Texto en dos formatos: csv `01/03/2025 05:37 a. m.` y xlsx `2025-03-01 05:37:12` | `datetime` (`inicio`) |
| `main_duracion_mm_ss` | `mm:ss` en el csv y `hh:mm:ss` en el xlsx | Segundos |
| `main_intervalo_de_ordeno_hh_mm` | Los de 24 h o más están guardados como fecha de Excel | Horas |
| `estado_patada`, `_incompleto`, `_pezones_no_encontrados`, `_amd_pezon` | Texto con lista de cuartos (`DI,TD`, `Todos`); el nulo significa «no ocurrió» | Banderas 0/1 por cuarto |
| `estado_numero_de_ordeno`, `estado_amd_ubre` | `float` solo por tener nulos | Sin cambio |
| `id_vaca`, `misc_eo_po` | Hashes (identificadores, no cantidades) | Texto |

## 1.3 Características del dataset

- **Unidad:** una visita de una vaca a una máquina. Tres máquinas con carga parecida (91 a 98 mil ordeños cada una) y un cuarto valor con 2 registros.
- **Vacas:** 207, con mediana de 1,264 sesiones (mínimo 2, máximo 3,569; tres con menos de 30). El hato crece de 34 vacas activas por mes (julio de 2022) a 182 (enero de 2026) con las mismas máquinas.
- **Columnas:** identificación y resumen (`main_*`), estado y eventos (`estado_*`), cinco mediciones por cuarto (producción, flujo medio y máximo, sangre, conductividad; sufijos `_di`, `_dd`, `_ti`, `_td`) y contexto (`misc_*`: máquina, destino de la leche, razón de la desviación).
- **No existe:** hora de salida (solo inicio y duración), fecha de parto, etiquetas de salud ni hora de llegada a la fila. El destino de la leche es Tanque en 94 % de los casos.

**Estadística básica** (280,697 ordeños; los estadísticos se calculan sobre los valores válidos):

| Variable | % nulos | Media | Desv. est. | Mín | Q1 | Mediana | Q3 | Máx |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Producción por ordeño (kg) | 0.0 | 14.82 | 5.34 | 0.00 | 11.07 | 14.29 | 18.10 | 78.17 |
| Duración (min) | 0.0 | 8.10 | 2.58 | 0.38 | 6.25 | 7.60 | 9.40 | 55.80 |
| Intervalo desde el ordeño anterior (h) | 0.2 | 9.92 | 3.86 | 0.01 | 7.41 | 9.21 | 11.64 | 47.92 |
| Número de ordeño del día | 0.0 | 1.84 | 0.84 | 1 | 1 | 2 | 2 | 6 |
| Flujo medio (kg/min, sin ceros) | 0.1 | 1.19 | 0.32 | 0.06 | 0.96 | 1.18 | 1.40 | 4.62 |
| Conductividad media (mS/cm, sin ceros) | 0.1 | 4.95 | 0.48 | 0.40 | 4.64 | 4.93 | 5.23 | 11.22 |
| MDI | 8.4 | 1.23 | 0.42 | 0.71 | 1.06 | 1.10 | 1.18 | 15.36 |
| Sangre máxima entre cuartos (ppm) | 0.0 | 52.55 | 386.39 | 0 | 0 | 0 | 0 | 8,000 |
| Producción cuarto DI (kg) | 1.9 | 3.51 | 1.58 | 0.00 | 2.49 | 3.38 | 4.42 | 39.84 |
| Producción cuarto DD (kg) | 2.6 | 3.64 | 1.58 | 0.00 | 2.61 | 3.50 | 4.55 | 38.71 |
| Producción cuarto TI (kg) | 1.6 | 4.12 | 2.04 | 0.00 | 2.88 | 4.02 | 5.33 | 31.42 |
| Producción cuarto TD (kg) | 3.9 | 3.93 | 2.03 | 0.00 | 2.70 | 3.85 | 5.12 | 39.06 |

La producción, la duración y el intervalo son asimétricos a la derecha: la media supera a la mediana en 3.7, 6.5 y 7.8 %. La sangre es 0 en más del 75 % de los ordeños pero llega a 8,000 ppm. El número de ordeño del día es casi siempre de 1 a 3 (en 45.5 % de los días-vaca hay 2 ordeños y en 36.4 % hay 3). Los mínimos de 0 en la producción por cuarto son cuartos sin leche. Los eventos son variables 0/1: patada 16.1 %, incompleto 11.9 %, pezón no encontrado 11.8 % y AMD de pezón 9.3 % de los ordeños.

## 1.4 ¿Los datos satisfacen los requerimientos del proyecto?

| Requerimiento | Estado | Evidencia |
| :---- | :---- | :---- |
| Identificador único de vaca en cada observación, clima e histórico de producción | **Cumple** | `id_vaca` sin nulos (207 vacas); clima en el 100 % de las sesiones; historial desde julio de 2022 |
| Variables del proceso de ordeño y del ambiente | **Cumple** | Duración, producción, flujo, conductividad, sangre, MDI, alertas por cuarto, destino de la leche; 13 variables de clima e índice THI |
| Clima preciso con la ubicación del rancho | **Parcial** | Celda de NASA POWER de ≈ 55 × 65 km, de exterior, con la hora desfasada una hora (se corrige) |
| Definir las clases de evento | **Parcial** | Hay cuatro banderas; el incompleto y el pezón no encontrado coinciden en 79 a 80 % |
| Probabilidad y 3 factores por alerta | **Pendiente (modelado)** | Hay variables candidatas: evento previo de la vaca, intervalo, cuartos sin leche |
| Permiso y protección de los datos | **Cumple / no verificable** | Vacas y máquinas anonimizadas; `Data/` y el mapa de equivalencias fuera del repositorio |
| Causas del incompleto revisadas por el socio formador | **Pendiente** | Preguntas en la sección 4 |

**Criterios de éxito.** El accuracy de «siempre sin evento» es 83.9 % para patada, 88.1 % para incompleto, 88.2 % para pezón no encontrado y 67.4 % para «al menos un evento». El criterio de 80 % se supera sin aprender nada en cada evento por separado, así que debe acompañarse de precisión, sensibilidad y F1, con validación por vaca y por tiempo.

**Supuestos y limitaciones frente a los datos.** (1) Los datos de ordeño abarcan desde julio de 2022, no solo 2025-2026; solo el concentrado y los rechazos son de 2025-2026. (2) El desbalance de eventos se confirma. (3) Los datos de producción son completos solo en parte: hay periodos secos y 28 % de ceros de conductividad en un cuarto. (4) Se asume una relación entre clima y eventos, pero la evidencia es débil. (5) El clima viene de reanálisis y observación, no de pronóstico; falta decidir qué usar al desplegar.

# **2. Reporte de exploración de los datos**

## 2.1 Visualizar y consultar los datos

- El 99.9 % de las visitas son ordeños. Uno de cada tres ordeños (32.6 %) tiene al menos un evento: patada 16.1 %, incompleto 11.9 %, pezón no encontrado 11.8 % y AMD de pezón 9.3 %.
- El incompleto y el pezón no encontrado alargan el ordeño unos 2 min (9.5 y 9.9 min contra 7.4).
- En 10.2 % de los ordeños algún cuarto da 0 kg; los traseros concentran el 75 % de esos casos.

![Producción diaria de las 30 vacas con más historia](figuras/nb_produccion_diaria_vacas.png)

*Cada punto gris es la producción de un día y la línea azul el promedio de 7 días: se ve la curva de lactancia (pico tras el parto y descenso gradual). Los puntos son dispersos porque dependen de cuántas veces vino la vaca (17.6 kg con un ordeño, 31.1 con dos y 42.0 con tres).*

**Ausencias de más de 40 días.** Hay 237 ausencias de más de 40 días (de 129 vacas) y la mediana dura 81 días, compatible con un periodo de secado. Al volver, las vacas producen unos 24 kg al día, como antes de irse (25 kg), y llegan a un pico de unos 55 kg en una mediana de 45 días. Estas ausencias explican los intervalos de más de 48 h y, como no hay fecha de parto, son la única forma de estimar la etapa de lactancia.

![Ausencias de más de 40 días](figuras/nb_ausencias_secado.png)

*Las zonas naranjas son los días sin ningún ordeño; son ejemplos de ausencias de más de 40 días con historia antes y después.*

## 2.2 Patrones y distribuciones

![Distribución de duración, producción e intervalo](figuras/nb_distribuciones.png)

- **Distribuciones:** producción, duración e intervalo son asimétricas a la derecha; en logaritmo quedan casi simétricas, lo que justifica transformarlas o usar métodos robustos.
- **Hora del día:** hay caídas de ordeños cerca de las 4, 12 y 20 h (pausas fijas por confirmar con el CAETEC). **Mes:** la mediana de producción por vaca-día va de 33.3 kg (septiembre) a 36.6 kg (febrero), unos 10 %, confundida con la etapa de lactancia. **Intervalo:** es el patrón más fuerte: la producción crece casi en línea recta con las horas desde el ordeño anterior (de unos 10 a 19 kg entre 4 y 15 h).

![Patrones por hora, día, mes e intervalo](figuras/nb_patrones_hora_mes.png)

- **Cuartos y máquinas:** los cuartos traseros producen más (4.1 y 4.0 kg contra 3.4 y 3.5 kg) porque se ordeñan más tiempo, con flujo y conductividad casi iguales, y se quedan sin leche más veces (4.6 y 4.7 % contra 1.5 y 1.6 %). La máquina no explica los eventos (V de Cramér ≤ 0.025).

![Producción por cuarto, cuartos sin leche y eventos por máquina](figuras/nb_cuartos_robots.png)

- **Eventos:** se concentran en pocas vacas (el 10 % de ellas acumula 38 % de las patadas) y se repiten: tras un evento en la sesión anterior la probabilidad de repetirlo es 39.9 % (patada), 30.9 % (incompleto) y 37.5 % (pezón), contra 11.5, 9.4 y 8.3 % sin evento previo. El incompleto se dispara si la vaca regresa antes de 4 h (32.7 % contra 10.7 a 12.5 %) y las vacas nuevas tienen más eventos en sus primeras semanas.

![Persistencia de los eventos](figuras/nb_persistencia.png)

- **Salud de la ubre:** MDI y conductividad máxima se mueven juntas (Spearman 0.55) y las visitas con alerta AMD de ubre tienen conductividad más alta (5.79 contra 5.03 mS/cm). Aparece sangre en 5.5 % de los ordeños, más en los cuartos delanteros, con picos por máquina (hasta 27.8 % en `7063e4b1` en el primer trimestre de 2024). Dentro de cada vaca, el 83 % produce menos con sangre detectada (−1.70 kg).

![MDI, conductividad y destino de la leche](figuras/nb_mdi_conductividad.png)

**Eventos y sangre en detalle.**

- **Eventos:** el incompleto y el pezón no encontrado coinciden en 79 y 80 % y afectan más a los cuartos traseros (5.0 a 5.6 % contra 1.9 a 2.5 % de los delanteros). El AMD de pezón varía entre máquinas (6.1 a 12.5 %). El incompleto subió de 9.4 % en 2022 a 12.7 % en 2025 y 40 vacas lo tienen en más del 20 % de sus ordeños. Los rechazos y las «echada sin ordeñar» solo se registran en 2025 (2.4 y 0.8 % de las visitas).
- **Sangre:** aparece en 5.5 % de los ordeños y en 204 de las 207 vacas, más en los cuartos delanteros (2.6 % en DI contra 1.1 % en TD); solo 1.1 % de los ordeños pasan de 1,000 ppm. La leche se desvía según la cantidad: con más de 1,000 ppm el 78.9 % va a `Divert 3` (0.5 % sin sangre), mientras que con conductividad ≥ 6.5 (2.7 % de los ordeños) sube el drenaje (12.7 % contra 3.6 %). La conductividad alta subió de 1.9 % en 2022 a 3.0 % en 2025 y se concentra en pocas vacas (12 con más del 10 % de sus ordeños).
- **Por máquina:** la sangre tiene picos de hasta 27.8 % de los ordeños en `7063e4b1` en el primer trimestre de 2024, y las mismas vacas dan 8.4, 2.9 y 6.4 % según la máquina, así que conviene revisar esos sensores con el cliente.

![Eventos del ordeño](figuras/nb_eventos.png)

![Sangre en el tiempo por máquina](figuras/nb_sangre_tiempo.png)

- **Correlaciones:** las más fuertes son producción con intervalo (0.49), duración con flujo (−0.39) y producción con incompleto (−0.33). No hay multicolinealidad grave (VIF máximo 2.84) y el PCA necesita 4 componentes para el 80 % de la varianza.

## 2.3 Problema de cola

El dataset no registra la llegada de la vaca a la fila ni la espera, solo el inicio y la duración del ordeño, así que la cola solo se puede estimar.

| Indicador | Resultado |
| :---- | :---- |
| Relevos pegados (la siguiente vaca entra a 2 min o menos de que termine la anterior) | De 21 % (2022) a 80 % (2026) |
| Ocupación de las máquinas | De 13 % a 71 % |
| Carga actual por máquina (últimos 3 meses) | 121 a 126 ordeños al día, 67 a 73 % de ocupación, hueco mediano de 0.5 a 0.6 min |
| Espera estimada (llegadas al azar) | De 8.8 a 12.9 min por vaca |
| Holgura | Solo las pausas de las 4:00 y las 20:00 |

![Ocupación de las máquinas y relevos pegados](figuras/nb_cola_ocupacion.png)

Por vaca, 14,046 ordeños superan el percentil 95 de duración de su propia vaca y 7,449 de ellos (53 %) además tienen producción o flujo bajos: son los candidatos a anomalía que ocupan la máquina sin producir. Falta la hora de llegada a la fila (preguntar al CAETEC si existe algún registro) para confirmar la espera.

## 2.4 Datos del clima integrados

Cada sesión se unió con la hora de NASA POWER (100 % de cobertura). La serie se corre una hora, porque la fila *t* describe el clima de *t* + 1 h, y se calculó el THI = (1.8 T + 32) − (0.55 − 0.0055 · HR) · (1.8 T − 26).

- **Matriz de correlación (Spearman) entre el ordeño y el clima de la hora de inicio:** el clima casi no se relaciona con el ordeño (|r| ≤ 0.07 con la producción, la duración, el flujo, la conductividad y los eventos de patada o incompleto; con las 13 variables climáticas, la correlación de cada evento es de 0.03 o menos). Lo que pesa es el propio proceso: la producción sube con las horas desde el ordeño anterior (0.54) y con la duración (0.30) y baja en los ordeños incompletos (−0.32); el flujo medio cae al alargarse el ordeño (−0.48). El número de ordeño del día correlaciona con la temperatura (0.34), probablemente por la hora del día y no por el calor, y temperatura y THI son casi la misma variable (0.99), así que basta usar una.

![Matriz de correlación entre el ordeño y el clima](figuras/nb_correlacion_clima.png)

- La producción relativa de cada vaca baja de 1.0 a 1.1 puntos porcentuales en días calurosos (p < 0.01 con THI máximo ≥ 68, 70 y 72); la hora de visita no cambia. La patada sube con el THI de la hora (15.1 % con THI < 55 a 17.6 % con THI ≥ 68); incompleto y pezón no.
- El clima diario tiene mucha redundancia: tres componentes explican 86.6 % (nivel térmico, nubosidad o amplitud, y viento).

![Producción relativa, hora de visita y eventos según el THI](figuras/nb_thi.png)

## 2.5 Hipótesis iniciales

| # | Hipótesis | Evidencia | Estado |
| :---- | :---- | :---- | :---- |
| H1 | Con THI alto baja la producción | −1.0 a −1.1 puntos porcentuales (p < 0.01) | Débil pero detectable |
| H2 | El calor cambia la hora de las visitas | 23.8 % contra 23.6 % entre 0 y 5 h | No sustentada |
| H3 | Los eventos dependen de la máquina y del cuarto | Máquina: efecto despreciable; cuarto en 0 kg en 71 % de incompletos y 87 % de pezones | Parcial |
| H4 | Los eventos se concentran en pocas vacas | El 10 % de las vacas acumula 31 a 38 % de cada evento | Sustentada |
| H5 | La conductividad antecede a las anomalías | Tasa de incompletos en U: 21.2 %, ≈ 10 %, 12.3 % por decil | Parcial (se mide en la misma sesión) |
| H6 | Los intervalos cortos se asocian a eventos | Incompleto 32.7 % con intervalo < 4 h | Sustentada (causalidad incierta) |
| H7 | Hay estacionalidad anual | Variación de ≈ 10 % entre septiembre y febrero | Parcial (confundida con lactancia) |
| H8 | Un evento anticipa el siguiente | 39.9 / 30.9 / 37.5 % contra 11.5 / 9.4 / 8.3 % | Sustentada |
| H9 | Las vacas nuevas tienen más eventos | Incompleto de 19 a 23 % (días 7 a 29) contra 11.6 % (≥ 240 días) | Parcial |
| H10 | MDI, conductividad y AMD señalan un mismo problema de ubre | Spearman 0.55; 5.79 contra 5.03 mS/cm | Sustentada como señal |
| H11 | Hay un problema de cola | Ocupación de 13 % a 71 %, relevos pegados de 21 % a 80 % | Sustentada con reservas |
| H12 | El concentrado se asocia con la producción | r = 0.60 el mismo día; 0.23 con un día de desfase | Parcial (asociación, no causal) |

# **3. Reporte de la calidad de los datos**

## 3.1 Análisis de la calidad

| Problema | Magnitud | Tratamiento |
| :---- | :---- | :---- |
| Sesiones repetidas entre csv y xlsx | 5,302 registros (formatos distintos) | Se quitan del csv, se conserva el xlsx |
| Duplicados exactos | 2 pares de rechazos | Eliminar |
| Columnas vacías o constantes | `estado_rcs_1000_celulas_ml` (100 % nula), `misc_usuario`, `misc_programa_de_lavado` (94.4 % nula) | Eliminar |
| Nulos que significan «no ocurrió» | Alertas: 84 a 91 % de nulos | Convertir a 0/1, no imputar |
| Nulo estructural | 216 visitas que no fueron ordeño | Separar; no imputar |
| Columnas de una sola fuente | `main_intervalo_de_ordeno_hh_mm`, `contadores_*`, `estado_mdi` (8.4 % de nulos) | Calcular el intervalo con la hora de inicio |
| Conductividad igual a 0 (imposible) | 1.7, 3.2, 8.3 y 28.3 % en DI, DD, TI y TD; en TD, el 89 % de sus ceros viene de la máquina `b96e85be` (sensor averiado) | Faltante; reportar al cliente |
| Cero de producción de un cuarto | 10.2 % de los ordeños | Dato real (cuarto sin leche); no imputar |
| Ordeños con 0 kg | 164, de los cuales 149 con todos los pezones no encontrados | Conservar y marcar |
| Total distinto de la suma de cuartos | 9 ordeños (> 0.05 kg) | Marcar |
| Intervalos de más de 48 h | 420 (regreso tras un periodo seco) | Faltante, con bandera |

![Porcentaje de nulos por columna](figuras/nb_nulos.png)

## 3.2 Datos atípicos

| Variable | Límites (1.5 IQR) | Atípicos | Extremos |
| :---- | :---- | ---: | :---- |
| Producción (kg) | 0.52 a 28.65 | 3,847 (1.37 %) | Máximo 78 kg; 55 ordeños > 40 kg |
| Duración (min) | 1.52 a 14.12 | 7,683 (2.74 %) | Máximo 55.8 min; 31 ordeños > 30 min |
| Intervalo (h) | 1.07 a 17.99 | 10,173 (3.63 %) | 420 de más de 48 h |

Otros atípicos: sangre de hasta 8,000 ppm (solo 1.1 % pasa de 1,000), MDI de hasta 15.36, 14,046 ordeños largos para su vaca (7,449 con producción o flujo bajos) y picos por máquina (caída de ordeños de `7063e4b1` en marzo de 2024). **No se eliminan:** pueden ser justo las anomalías a detectar; se transforman (logaritmo) o se usan métodos robustos.

## 3.3 Métodos de limpieza recomendados

Al final de cada acción se indica si el notebook ya la aplica o si queda para la preparación de datos.

1. Convertir fechas, duración e intervalo a `datetime` y número (los intervalos de Excel se leen como fecha). Estado: aplicada.
2. Quitar las 5,302 sesiones repetidas entre csv y xlsx y los 2 duplicados exactos. Estado: aplicada.
3. Eliminar columnas vacías o constantes. Estado: aplicada.
4. Separar los 216 rechazos y echadas sin ordeñar; dejar fuera la máquina con 2 registros. Estado: aplicada.
5. Convertir las alertas de texto a banderas 0/1 por cuarto. Estado: aplicada.
6. Conductividad 0 → faltante. En el cuarto trasero derecho de `b96e85be`, calcular la razón de conductividad solo con cuartos válidos o excluir ese cuarto. Estado: parcial.
7. Intervalos de más de 48 h → NaN y bandera «regreso tras periodo seco». Estado: parcial.
8. No imputar cuartos sin leche ni nulos estructurales; MDI y sensores: manejo de nulos del modelo o imputación por vaca, sin imputación múltiple ciega. Estado: parcial.
9. Conservar y marcar los 164 ordeños con 0 kg y los 9 con suma de cuartos distinta. Estado: por aplicar.
10. No eliminar atípicos: logaritmo o métodos robustos. Estado: por aplicar.
11. Rezagos y promedios móviles por vaca con desplazamiento de un ordeño (sin filtrar el presente), reiniciando tras ausencias de 45 días o más. Estado: por aplicar.
12. Excluir del historial por vaca a las que tienen menos de 30 registros. Estado: por aplicar.
13. Clima: desfase de una hora, THI por hora, máximo diario y rezagos. Estado: parcial.
14. Definir las clases (prioridad o etiquetas múltiples) antes de construir el objetivo. Estado: por aplicar.
15. Normalizar o marcar la sangre por máquina (picos de 27.8 % y máximos de 8,000 ppm). Estado: por aplicar.

# **4. Implicaciones para el modelado y pendientes**

1. **Definir las clases con cuidado.** El incompleto y el pezón no encontrado casi coinciden; no usar uno para predecir el otro.
2. **Evitar la fuga de información.** Duración, producción, conductividad, flujo, cuartos en 0 kg, destino de la leche y razón de desviación se miden en la misma sesión y pueden ser consecuencia del evento; para una alerta temprana usar el intervalo previo, el historial de la vaca y el clima.
3. **Aprovechar el historial de la vaca** (evento previo, tasa propia): es la señal más clara.
4. **Validar por vaca y por tiempo**, con precisión, sensibilidad y F1 además del accuracy.
5. **Clima como variable de contexto** (THI máximo del día, rezagos, nubosidad y viento), sin esperar que sea la señal principal.

**Pendientes con el socio formador:** las pausas de las 4, 12 y 20 h; el sensor de conductividad del cuarto trasero derecho de `b96e85be`; los picos de sangre y la caída de `7063e4b1` en 2024; el umbral de conductividad (6.5 mS/cm por confirmar); la definición de `incomplete`, `teat_missing` y `kick`; la fecha de parto; la unidad del concentrado; y si existe un registro de llegada a la fila. No hay fecha de parto ni etiquetas de salud: la lactancia se aproxima por brechas de registro.
