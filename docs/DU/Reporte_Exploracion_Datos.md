# Reporte de exploración de datos

Fase *Data Understanding* (CRISP-DM). Cubre cuatro puntos: visualizar y consultar los datos, visualizar patrones y distribuciones, integrar datos del clima e identificar hipótesis iniciales. La elección de la fuente de clima está en [Investigacion_Clima.md](Investigacion_Clima.md).

**Objetivo de negocio:** detectar de forma temprana anomalías en el ordeño. **Objetivo de minería de datos:** clasificar el tipo de evento de ordeño (patada, ordeño incompleto, pezón no encontrado).

> Los resultados son exploratorios: describen asociaciones y no demuestran causas. Todos los identificadores de vaca se muestran como alias (`V001`, `V002`, …) asignados por fecha del primer registro.

## 1. Datos y método

| Elemento | Detalle |
|---|---|
| Tabla base | Eventos de ordeño normalizados: **280,699 sesiones de 207 vacas**, del 13/07/2022 al 28/01/2026. Una fila es una sesión de ordeño de una vaca en un robot (VMS) |
| Clima | NASA POWER horario, 100 % de las sesiones con clima. Se corre +1 hora para alinearlo (ver [Investigacion_Clima.md](Investigacion_Clima.md)) |
| Código | [`src/DU/exploracion_datos.py`](../../src/DU/exploracion_datos.py) genera todas las figuras y los indicadores; [`src/DU/clima_comparacion.py`](../../src/DU/clima_comparacion.py) descarga el clima |
| Datos | Fuera del repositorio (`Data/`), según la política de uso de datos |

**Variables.** Por sesión: producción (`milk_kg`), duración, intervalo desde el ordeño anterior, número de sesión del día, y por cuarto de la ubre (DI, DD, TI, TD) la producción, el flujo, la sangre y la conductividad; además `mdi`, el robot (`machine`), el destino de la leche y tres banderas binarias: `kick` (patada), `incomplete` (ordeño incompleto) y `teat_missing` (pezón no encontrado).

## 2. Visualizar y consultar los datos

### 2.1 Resumen de volumen y calidad

| Indicador | Resultado |
|---|---|
| Sesiones por vaca | Mediana 1,264 (mínimo 2, máximo 3,569); 3 vacas con menos de 30 sesiones |
| Sesiones por robot | VMS 1: 97,643 · VMS 2: 91,649 · VMS 3: 91,405 |
| Destino de la leche | Tanque 94.4 % · Drenaje 3.7 % · Divert 3 1.9 % (Divert 1 y 2: 7 sesiones) |
| Duplicados (misma vaca y hora) | 0 |
| Producción ≤ 0 kg / > 40 kg | 166 / 55 sesiones |
| Duración > 30 min | 31 sesiones |
| Suma de cuartos distinta del total (> 0.2 kg) | 9 sesiones |
| Brechas de 45 días o más sin registros | 249 brechas en 134 vacas (no son producción cero: secado, parto o falta de captura) |
| Variable vacía | `rcs` (conteo de células somáticas): 100 % nula, no utilizable |
| Nulos en variables por cuarto | 1.6 % a 3.9 %; `mdi` 8.4 %; `interval_h` 0.6 % |

![Porcentaje de nulos por variable](figuras/nulos_por_variable.png)

![Sesiones de ordeño por vaca y mes](figuras/cobertura_por_vaca.png)

**Ceros que en realidad son faltantes.** La conductividad de la leche no puede ser 0. Hay ceros en el 1.7 % (DI), 3.2 % (DD), 8.3 % (TI) y 28.3 % (TD) de las sesiones. En TD casi todos vienen de un solo robot: **VMS 1 tiene conductividad TD igual a 0 en el 72.3 % de sus sesiones** frente al 5.2 % y 4.3 % de VMS 2 y VMS 3, lo que apunta a un sensor averiado en ese cuarto y robot. En los demás casos el cero coincide con un cuarto que no dio leche (producción 0 kg en el 1.6 %, 1.5 %, 4.6 % y 4.7 % de las sesiones para DI, DD, TI y TD). Para análisis, la conductividad y el flujo en 0 se tratan como faltantes.

### 2.2 Consultas de ejemplo

```python
leche.groupby("machine")[["kick", "incomplete", "teat_missing"]].mean() * 100   # eventos por robot
leche[leche["incomplete"] == 1]["milk_kg"].median()                             # leche en ordeños incompletos
leche.groupby("animal_id")["kick"].mean().sort_values(ascending=False).head()   # vacas con más patadas
leche[(leche["interval_h"] < 4)]["incomplete"].mean()                           # incompletos con intervalo corto
```

| Consulta | Resultado |
|---|---|
| Mediana de leche en un ordeño incompleto frente a uno completo | 9.7 kg frente a 14.9 kg; duración mediana de 9.5 min frente a 7.4 min |
| Vacas con patada en más del 40 % de sus sesiones | 12 vacas (las dos primeras, `V183` y `V203`, superan el 85 %) |
| Sesiones con menos de 0.5 h desde el ordeño anterior | 893 (0.3 %); 16.8 % de ellas es incompleta |

## 3. Visualizar patrones y distribuciones

### 3.1 Distribuciones de las variables numéricas

![Distribuciones](figuras/eda_distribuciones.png)

| Variable | Mediana | Observación |
|---|---:|---|
| Producción por sesión (kg) | 14.29 | Forma de campana con cola derecha |
| Duración (min) | 7.60 | Sesgo a la derecha; casi no hay sesiones de menos de 4 min |
| Intervalo entre ordeños (h) | 9.01 | Tope en 24 h; hay un pico pequeño cerca de 0 h (sesiones repetidas en minutos) |
| Flujo medio por cuarto (kg/min) | 1.18 | Valores discretos por la resolución del sensor |
| Conductividad media (mS/cm) | 4.93 | Casi simétrica, dentro del rango fisiológico |
| MDI | 1.10 | Muy concentrada en 1.0–1.2 con una cola larga y un tope en 2.5 |

### 3.2 Patrones temporales

![Patrones temporales](figuras/eda_patrones_temporales.png)

- **Hora del día:** las visitas se reparten durante las 24 horas (el voluntario no tiene horario fijo), con caídas marcadas a las **04 h (2.2 %), 12–13 h (≈ 3.2 %) y 20 h (2.3 %)** frente a ≈ 4.8 % en las horas pico. Podrían corresponder a ciclos de limpieza del robot; se debe confirmar con el socio formador.
- **Día de la semana:** sin diferencias (14.1 % a 14.5 % por día).
- **Sesiones por vaca-día:** mediana 2, media 2.42, máximo 6. Producción mediana por vaca-día de 34.9 kg.
- **Mes a mes:** la producción por vaca-día se mueve entre 32.3 y 38.6 kg sin una estacionalidad anual clara. El hato crece de 37 a 172 vacas activas, por lo que la composición de lactancias cambia con el tiempo y confunde cualquier tendencia.

### 3.3 Curva de lactancia aproximada

![Curva de lactancia](figuras/eda_lactancia.png)

Con 249 episodios que reinician tras una brecha de 45 días o más, la mediana alcanza su pico hacia el **día 39 con 50.4 kg por vaca-día** y baja a 34.1 kg hacia el día 250. El rango intercuartil es amplio (± 8 kg en el pico), por lo que la etapa de lactancia explica mucha variación entre vacas. No hay fecha de parto, así que los días se miden desde el regreso al ordeño.

### 3.4 Cuartos y robots

![Cuartos y robots](figuras/eda_cuartos_y_vms.png)

- **Cuartos traseros:** producen más leche cuando dan leche (TI 4.32 kg y TD 4.13 kg frente a DI 3.57 kg y DD 3.70 kg), pero tienen 3 veces más sesiones con 0 kg (4.6 % y 4.7 % frente a 1.6 % y 1.5 %).
- **Robots:** las tasas de eventos son parecidas. Patada: 15.8 %, 15.2 % y 17.3 % (VMS 1, 2 y 3). Ordeño incompleto: 12.1 %, 12.2 % y 11.6 %. Pezón no encontrado: 11.6 %, 11.7 % y 12.0 %. La prueba chi-cuadrada sale significativa en patada y en incompleto por el tamaño de la muestra, pero el tamaño del efecto es mínimo (V de Cramér de 0.025 y 0.008).

### 3.5 Eventos objetivo

![Eventos](figuras/eda_eventos.png)

| Evento | Prevalencia | % de los eventos concentrado en el 10 % de las vacas con más eventos |
|---|---:|---:|
| Patada | 16.1 % | 38.0 % |
| Ordeño incompleto | 11.9 % | 30.8 % |
| Pezón no encontrado | 11.8 % | 36.9 % |
| Alguno de los tres | 26.4 % | — |

- **Desbalance moderado:** ninguna clase baja del 11 %. El criterio de accuracy ≥ 70 % se supera sin esfuerzo con una clase que domina (73.6 % de sesiones sin ningún evento), por lo que habrá que reportar también precisión, exhaustividad y F1 por clase.
- **`incomplete` y `teat_missing` casi son el mismo evento:** el 79 % de los ordeños incompletos tiene pezón no encontrado, el 80 % de los pezones no encontrados es incompleto y su correlación de Spearman es de **0.77**. Usar uno como variable para predecir el otro sería fuga de información.
- **Las patadas son otro fenómeno:** solo el 21 % de las patadas coincide con un ordeño incompleto.
- **Concentración por vaca:** el 10 % de las vacas aporta alrededor de un tercio de cada evento. La identidad de la vaca es una señal fuerte, y es un riesgo si el modelo debe generalizar a vacas nuevas.

### 3.6 Correlaciones

![Correlaciones](figuras/eda_correlaciones.png)

La producción se asocia sobre todo con el intervalo desde el ordeño anterior (Spearman 0.53), y también con la duración (0.30), la conductividad (0.30) y el flujo (0.27). Los ordeños incompletos producen menos (−0.32) y duran más (0.24). El intervalo no se correlaciona linealmente con la bandera de incompleto (0.00), pero la relación no es lineal (ver H6 en la sección 5).

## 4. Datos del clima integrados

Se unió cada sesión con la hora de clima de NASA POWER (cobertura del 100 %). Se calculó el índice temperatura-humedad `THI = (1.8 T + 32) − (0.55 − 0.0055 HR) · (1.8 T − 26)`. En el periodo, el 21.2 % de las sesiones ocurre con THI ≥ 68 y hay 711 de 1,299 días con THI máximo ≥ 70.

![Clima y THI](figuras/eda_clima_thi.png)

Para quitar las diferencias entre vacas y etapas de lactancia, la producción de cada vaca-día se divide entre la mediana de sus 28 registros anteriores ("producción relativa").

## 5. Hipótesis iniciales

| # | Hipótesis | Evidencia exploratoria | Estado |
|---|---|---|---|
| H1 | Con THI alto baja la producción por vaca | La mediana de producción relativa es casi plana entre grupos de THI máximo (0.996 a 1.015). A nivel de hato, la correlación con el THI medio es débil y negativa: Spearman de −0.12 a −0.14 con rezagos de 0 a 5 días (p < 0.001) | **Débil o no sustentada.** Hay poco contraste (más de la mitad de los días ya tiene THI ≥ 70) y el THI es de exterior y de celda gruesa |
| H2 | El calor cambia la frecuencia o la hora de las visitas | El reparto por hora es casi idéntico en días frescos y calurosos: 23.6 % frente a 23.8 % de sesiones entre 00 y 05 h y 25.0 % frente a 25.2 % entre 12 y 17 h | **No sustentada** con este clima |
| H3 | La proporción de eventos depende del robot y del cuarto | Diferencias pequeñas entre robots (V de Cramér ≤ 0.025); VMS 3 tiene más patadas (17.3 %). Los cuartos traseros tienen 3 veces más sesiones sin leche | **Parcial:** efecto del robot mínimo; el del cuarto es real |
| H4 | Las patadas se concentran en pocas vacas | El 10 % de las vacas aporta el 38 % de las patadas; 12 vacas patean en más del 40 % de sus sesiones | **Sustentada** |
| H5 | La conductividad alta antecede a las anomalías | La tasa de incompletos tiene forma de U según el decil de conductividad: 21.2 % en el decil más bajo, 9.6 % a 10 % en el centro y 12.3 % en el más alto. El extremo bajo es más fuerte que el alto | **Parcial.** Cuidado: la conductividad se mide en la misma sesión, por lo que puede ser consecuencia del incompleto y no un predictor |
| H6 | Los intervalos extremos se asocian a eventos | Incompletos: 25.9 % con intervalo menor a 4 h, 10.9 % a 12.4 % entre 4 y 16 h, 15.8 % con 16 h o más | **Sustentada.** El sentido causal no está claro: un intervalo corto puede ser un reintento tras un ordeño incompleto |
| H7 | La producción tiene estacionalidad anual y tendencia | Sin patrón anual claro en 3.5 años; la lactancia (pico al día 39) y el crecimiento del hato dominan | **No concluyente** |

Hallazgo adicional: la **tasa de patadas sube con el THI de la hora de la sesión**, de 15.1 % con THI menor a 55 a 17.6 % con THI de 68 o más (+2.5 puntos porcentuales). Está confundida con la hora del día (las patadas son más frecuentes a mediodía y tarde, 16.3 % a 17.6 % entre 10 y 20 h frente a 14.6 % a 15.7 % de madrugada); se debe controlar antes de afirmarlo. Los ordeños incompletos y los pezones no encontrados no muestran relación con el THI.

![Eventos por intervalo y conductividad](figuras/eda_intervalo_conductividad.png)

## 6. Implicaciones para el modelado

1. **Definir las etiquetas con cuidado.** `incomplete` y `teat_missing` están casi fusionados; conviene decidir si son una o dos clases y excluir la información de una al predecir la otra.
2. **Variables medidas en la misma sesión** (conductividad, duración, producción) pueden ser consecuencia del evento; para una alerta temprana hay que usar solo lo disponible antes del evento (intervalo previo, historial de la vaca, clima).
3. **Validar por vaca**, no por sesión: la identidad de la vaca explica gran parte de los eventos y una validación aleatoria sobreestimaría el desempeño.
4. **Tratar los ceros de conductividad y flujo como faltantes**, y considerar la avería del sensor TD en VMS 1.
5. **Clima:** el efecto aparente sobre producción y visitas es pequeño. Conviene probarlo como variable de contexto (THI máximo del día y rezagos), sin esperar que sea la señal principal.
6. **No imputar los cuartos sin leche:** esa leche no existió; el modelo debe saber cuántos cuartos ordeña cada vaca.

## 7. Limitaciones y pendientes

- La unidad de consumo de concentrado no está declarada, y el consumo solo existe desde 28/01/2025; no se exploró aquí.
- No hay fecha de parto ni etiquetas de salud: la lactancia se aproxima por brechas de registro.
- El clima es de exterior y de una celda de ≈ 50 × 60 km; no mide el microclima del establo.
- Falta confirmar con el socio formador la interpretación de los huecos de las 04, 12–13 y 20 h, el sensor TD de VMS 1 y la definición exacta de `incomplete`, `teat_missing` y `kick`.
- Falta decidir si el modelo se despliega con pronóstico o con clima observado.

## 8. Cómo reproducirlo

```bash
python src/DU/clima_comparacion.py      # descarga y compara el clima (usa red)
python src/DU/exploracion_datos.py      # genera figuras e indicadores
```

Ambos scripts leen los datos desde la carpeta de tablas normalizadas (`--tablas`), escriben las figuras en `docs/DU/figuras/` y guardan los datos intermedios (clima, indicadores en JSON) en `Data/`, fuera del control de versiones. Requieren `numpy`, `pandas`, `matplotlib`, `scipy` y `requests`.
