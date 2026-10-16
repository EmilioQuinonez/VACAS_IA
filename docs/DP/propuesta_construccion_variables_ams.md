# Propuesta de Construcción de Variables, Clasificación de Eventos y Modelado en AMS

**Proyecto:** Detección de Anomalías de Ordeño y Clasificación de Eventos en DeLaval VMS  
**Objetivo de Negocio:** Identificar problemas de cola mediante la detección de anomalías en el ordeñamiento.  
**Objetivo de Ciencia de Datos:** Clasificar los posibles tipos de evento en el ordeño e identificar los 3 factores con >80% de impacto predictivo.

---

## 1. Construcción de Nuevas Variables (Feature Engineering) y Justificación Científica

### A. Factores Externos (Clima, Moscas y Fallas del Robot)

#### 1. Índice de Estrés Térmico (THI) y Efectos Retardados (THI_lag1, THI_lag2, THI_lag3)
* **Fórmula:**
  $$THI_{max} = (1.8 \cdot T_{max} + 32) - [(0.55 - 0.0055 \cdot RH) \cdot (1.8 \cdot T_{max} - 26)]$$
* **Variables creadas:**
  * `THI_max_dia`: THI máximo del día del ordeño.
  * `THI_lag1`, `THI_lag2`, `THI_lag3`: Promedio de $THI_{max}$ de 1, 2 y 3 días previos.
  * `Categoria_THI`: Sin estrés ($<72$), Leve ($72-77$), Moderado ($78-81$), Severo ($\ge 82$).
* **Justificación científica:**
  * El impacto del estrés térmico en la producción de leche no es inmediato, sino que presenta un desfase fisiológico de 1 a 3 días previos, provocando caídas de $0.18\text{ kg}$ a $0.40\text{ kg}$ por unidad de $THI$ incrementado.
  * En cambio, el $THI$ del mismo día altera de forma directa e inmediata el comportamiento del ganado: disminuye el tiempo de pastoreo/rumia e incrementa el tiempo de pie y las visitas voluntarias al robot buscando sombra o alimento concentrado.
* **Fuentes:** 
  * *Impact of heat stress on dairy cow rumination, milking frequency, milk yield and quality in a pasture-based automatic milking system.pdf*
  * *The effects of heat stress on milk production and the grazing behavior of dairy Holstein cows milked by an automatic milking system.pdf*
  * *Characterization of failed milking events in a voluntary milking system.pdf*

#### 2. Proxy de Presión de Moscas y Molestia Ambiental (`Fly_Pressure_Proxy`)
* **Lógica / Fórmula:**
  $$\text{Fly\_Pressure\_Proxy} = \text{Flag\_Tarde (13:00-17:00 h)} \times T_{ambiente} \times \text{Tasa\_Historica\_Patadas\_Vaca}$$
* **Variables creadas:**
  * `Flag_Tarde`: Variable binaria ($1$ si el ordeño ocurrió por la tarde, $0$ si fue en la mañana/noche).
  * `Fly_Risk_Score`: Combinación de temperatura ambiental ($>15^\circ\text{C}$) y hora del día.
* **Justificación científica:**
  * Las moscas picadoras (*Stomoxys calcitrans* y *Haematobia irritans*) atacan predominantemente por la tarde ($1.5$ veces más presencia de moscas en patas por la tarde) y con temperaturas $>15^\circ\text{C}$.
  * Las moscas detonan respuestas defensivas motoras inmediatas (patadas/pisotones de patas traseras, sacudidas de cabeza, coletazos y espasmos cutáneos). El pisotón de patas aumenta un $26\%$ por la tarde y se reduce un $39\%$ al usar mallas protectoras. Esto genera agitación durante el acoplamiento pezonera-pezón en el VMS.
* **Fuentes:**
  * *Effects of mesh leggings on fly pressure and fly avoidance behaviors of pastured dairy cows.pdf*
  * *The Defensive Behaviors and Milk Production of Pastured Dairy Cattle in Response to Stable Flies, Horn Flies, and Face Flies*

#### 3. Categorización de Ordeños No Exitosos / Fallidos (`FME_Category`)
* **Lógica de construcción** (a partir de `data_example.txt`):
  * **`FME_Conexion` (CA / CT):** Si `estado_pezones_no_encontrados` $>0$ o el tiempo de acoplamiento excede el umbral recomendado.
  * **`FME_Leche_Muerta` (DT - Dead Milk Time):** Si las pezoneras se conectan pero el flujo inicial es cero (`producciones_kg_*` $= 0$ o `media_de_los_flujos` $= 0$).
  * **`FME_Incompleto`:** Si `estado_incompleto` es verdadero o la producción por cuarto es $\le 50\%$ de lo esperado.
* **Justificación científica:**
  * Los eventos de ordeño fallido (*Failed Milking Events* - FME) representan un grave problema de eficiencia en AMS.
  * El $35.1\%$ de las fallas se deben a tiempo de conexión prolongado (CT), el $21.9\%$ a múltiples intentos fallidos (CA) y el $35.4\%$ a leche muerta (DT).
  * La presencia de mastitis incrementa $6.7$ veces las probabilidades de sufrir un FME ($\text{OR} = 6.70$).
* **Fuentes:**
  * *Characterization of failed milking events in a voluntary milking system.pdf*
  * *Frontiers | Association between health status and the temporal dynamics of milking behavior and performance in early-lactation cows in an automatic milking system*

---

## 1. B. Factores Internos (Salud, Conducta del Animal y Calidad de Leche)

#### 1. Índice de Severidad de Patadas / Desacoples (`Patada_Severity_Index`)
* **Lógica de construcción** (a partir de `data_example.txt`):
  * `Num_Cuartos_Patada`: Conteo de cuartos afectados en la columna `estado_patada` (ej. `"DI"` $= 1$, `"DI,TD"` $= 2$).
  * `Patada_Ratio_7d`: Proporción de ordeños con registro de patada en los últimos 7 días por vaca.
* **Justificación científica:**
  * Las patadas son la respuesta motora directa ante el dolor o la irritación (por insectos, edema de ubre o inflamación mamaria).
  * En el robot conducen al desacople brusco de las pezoneras (*kick-off*), interrupción repentina del flujo de leche e incompletitud del ordeño.
* **Fuentes:**
  * *Frontiers | Association between health status and the temporal dynamics of milking behavior and performance...*
  * *Multivariable time series classification for clinical mastitis detection and prediction in automated milking systems.pdf*

#### 2. Carga y Desviación de Sangre en Leche (`Sangre_PPM_Max` y `Sangre_Cuarto_Delta`)
* **Lógica de construcción** (a partir de `data_example.txt`):
  * `Sangre_PPM_Max`: Valor máximo de PPM entre los 4 cuartos (`sangre_ppm_di`, `dd`, `ti`, `td`).
  * `Sangre_Desviacion_Cuarto`: $\text{sangre\_ppm\_cuarto} - \text{media\_sangre\_vaca\_ordeño}$.
* **Justificación científica:**
  * La presencia de sangre en leche indica trauma tisular, ruptura microvascular o inflamación severa en el pezón/ubre, siendo una variable crítica para algoritmos de detección de anomalías y rechazo de leche.
* **Fuentes:**
  * *Multivariable time series classification for clinical mastitis detection and prediction...*
  * *Influence of Preprocessing Methods of Automated Milking Systems Data on Prediction of Mastitis...*

#### 3. Ratio y Media Móvil de Conductividad Eléctrica (`EC_Ratio`, `EC_MA_7d`, `EC_SD_7d`)
* **Lógica de construcción** (a partir de `data_example.txt`):
  * `EC_Ratio_Cuarto`: $\frac{\text{conductividad\_cuarto}}{\text{media\_conductividad\_4\_cuartos\_mismo\_ordeño}}$.
  * `EC_MA_7d`: Media móvil de conductividad por cuarto en los últimos 7 días (o 9-14 ordeños).
  * `EC_SD_7d`: Desviación estándar móvil de la conductividad por cuarto en los últimos 7 días.
* **Justificación científica:**
  * La infección mamaria altera la concentración iónica de la leche ($\text{Na}^+$ y $\text{Cl}^-$), incrementando fuertemente la conductividad eléctrica (EC) del cuarto afectado respecto a los sanos.
  * La combinación de EC con su media y desviación estándar móvil de 7 días permite detectar mastitis clínica y subclínica incluso 1 ordeño antes de la manifestación evidente.
* **Fuentes:**
  * *Multivariable time series classification for clinical mastitis detection and prediction...*
  * *Influence of Preprocessing Methods of Automated Milking Systems Data on Prediction of Mastitis...*

#### 4. Ratio de Eficiencia de Box y Tiempo de Permanencia (`Box_Time_to_Yield_Ratio`)
* **Lógica de construcción** (a partir de `data_example.txt`):
  $$\text{Box\_Time\_to\_Yield\_Ratio} = \frac{\text{main\_duracion\_mm\_ss (en minutos)}}{\text{main\_produccion\_kg}}$$
* **Justificación científica:**
  * Vacas enfermas o inquietas pasan más tiempo dentro del cajón del robot (*Box Time*) debido a retrasos en el acoplamiento o eyección lenta de leche, produciendo un volumen menor. Esto eleva dramáticamente la relación minutos/kg.
  * Velocidades de ordeño bajas ($<2.7\text{ kg/min}$) aumentan un $28\%$ la probabilidad de FME ($\text{OR} = 1.28$).
* **Fuentes:**
  * *Frontiers | Association between health status and the temporal dynamics of milking behavior...*
  * *Characterization of failed milking events in a voluntary milking system.pdf*

---

## 2. Clasificación de Tipos de Eventos de Ordeño (Objetivo de Ciencia de Datos)

### Estructura del Target ($Y$)
Opción Multiclase (una sola columna $Y$) recomendada para el entrenamiento inicial.

| Clase | Nombre del Evento | Criterio / Definición Lógica |
| :---: | :--- | :--- |
| **0** | **Ordeño Normal / Exitoso** | `estado_patada` $= 0$, sin sangre (`sangre_ppm` $= 0$), `EC_Ratio` $< 1.10$, producción esperada completa e intervalo regular ($8-14\text{ h}$). |
| **1** | **Interrupción por Conducta / Moscas** | `estado_patada` $> 0$ o desacople de pezonera (*kick-off*), sin alteraciones de conductividad ni sangre, alta correlación con horario vespertino y $THI$ elevado. |
| **2** | **Anomalía de Salud / Mastitis / Lesión** | Conductividad elevada (`EC_Ratio` $> 1.15$), `sangre_ppm` $> 0$, alarma RCS / MDI activada (`estado_mdi` o $\text{RCS} > 200\text{k cel/mL}$). |
| **3** | **Fallo Mecánico / Conexión (FME)** | `estado_pezones_no_encontrados` verdadero, tiempo/intentos de conexión excedidos (CA/CT) o leche muerta (DT con flujo 0). |

---

## 3. Los 3 Factores Principales en la Predicción (>80% de Importancia Acumulada)

De acuerdo con la literatura de modelos de ensamble (*Random Forest*, *XGBoost*, *Decision Trees*) aplicados a datos de robots AMS:

1. **Desviación Relativa de Conductividad Eléctrica por Cuarto (`EC_Ratio` / `EC_MA_7d`)**
   * *Peso predictivo:* Constituye el principal factor discriminante para diferenciar anomalías patológicas/sanitarias de problemas puramente mecánicos o conductuales.
2. **Registro e Intensidad de Patadas / Desacoples (`estado_patada` / `Num_Cuartos_Patada`)**
   * *Peso predictivo:* Indicador primario de la respuesta conductual/motora. Explica los desacoples prematuros y distingue la irritación por insectos/edema de ubre.
3. **Rendimiento e Intervalo de Ordeño (`producciones_kg`, `main_intervalo_de_ordeno_hh_mm`, `Box_Time_to_Yield_Ratio`)**
   * *Peso predictivo:* Métricas fisiológicas acumuladas que capturan el estado general de la vaca (desviaciones respecto a la media de 14 ordeños causadas por estrés térmico o enfermedades sistémicas).

---

## 4. Recomendaciones de Preprocesamiento y Metodología de Modelado

1. **Ventanas Temporales en Series de Tiempo (*Lagged Features*):**
   * Crear variables autorregresivas basadas en los últimos **7 a 9 ordeños anteriores** ($\sim 3$ días previos). Esta ventana ofrece la mayor precisión en la predicción de mastitis y eventos fallidos.
   * Suavizado de producción y velocidad de ordeño mediante media móvil de **14 sesiones previas**.
2. **Manejo de Valores Faltantes (Imputación):**
   * Los sensores de robots presentan un $\sim 30\%$ de datos faltantes.
   * Para modelos de ensamble, utilizar **MICE** (*Multiple Imputation with Chained Equations*) o *Simple Imputer*, or evaluar el análisis sobre casos completos complementarios.
3. **Rebalanceo de Clases (*Imbalanced Data*):**
   * Los ordeños anómalos o enfermos representan eventos raros ($<1-5\%$ del total de ordeños).
   * Aplicar técnicas de sobremuestreo/submuestreo combinado como **SVMSMOTE** o **SMOTEEN** en el conjunto de entrenamiento para mejorar la sensibilidad ($>82-95\%$) sin inflar los falsos positivos.
4. **División de Datos (*Train / Validation / Test*):**
   * Dividir estrictamente por **orden cronológico** (ejemplo: $80\%$ entrenamiento/validación, $20\%$ prueba final) para evitar fuga de datos (*data leakage*) entre ordeños consecutivos de una misma vaca.
