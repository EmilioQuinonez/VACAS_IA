# Manual de uso de la interfaz de predicción

La interfaz permite probar los tres modelos del proyecto (LSTM v5, regresión logística v4 y XGBoost v5) desde el navegador. Para un ordeño, cada modelo da:

- la **probabilidad** de cada uno de los 4 eventos: falla mecánica, salud, conducta y alguna anomalía;
- si esa probabilidad genera una **alerta**;
- la **variable que más aumenta el riesgo** en ese ordeño, explicada sin tecnicismos y con qué revisar.

La interfaz no entrena ni modifica los modelos ni los datos: usa los modelos guardados en el repositorio y el dataset integrado que ya está en la computadora.

## Contenido

1. [Requisitos](#1-requisitos)
2. [Cómo abrir la interfaz](#2-cómo-abrir-la-interfaz)
3. [La pantalla](#3-la-pantalla)
4. [Predecir una sesión nueva](#4-predecir-una-sesión-nueva)
5. [Predecir un ordeño del dataset](#5-predecir-un-ordeño-del-dataset)
6. [Cómo leer los resultados](#6-cómo-leer-los-resultados)
7. [Variables de la sesión y desempeño de los modelos](#7-variables-de-la-sesión-y-desempeño-de-los-modelos)
8. [Avisos y mensajes de error](#8-avisos-y-mensajes-de-error)
9. [Limitaciones](#9-limitaciones)
10. [Problemas comunes](#10-problemas-comunes)

## 1. Requisitos

- **Python 3** con las dependencias del proyecto, instaladas desde la raíz del repositorio:

  ```bash
  python3 -m pip install -r requirements.txt
  ```

  En Windows, usa `py` en lugar de `python3`.

- **El dataset integrado**, `Data/dataset_vacas_integrado.csv`, generado con los pasos de la sección *¿Cómo obtener y preparar los datos?* del [README](../../README.md) y con las variables de historial de `src/MODELING/LSTM/build_history_features.py`. Los datos no están en el repositorio: cada persona los genera en su computadora.
- **En Mac:** XGBoost necesita la librería OpenMP. Si no la tienes, instálala con `brew install libomp`.
- **Un navegador** actualizado (Chrome, Edge, Firefox o Safari).

## 2. Cómo abrir la interfaz

1. Abre una terminal en la raíz del repositorio.
2. Inicia el servidor:

   | Sistema | Comando |
   |---|---|
   | macOS y Linux | `python3 src/INTERFAZ/app.py` |
   | Windows | `py src/INTERFAZ/app.py` |

3. Espera unos 10 segundos, mientras se cargan el dataset y los modelos, hasta que la terminal muestre:

   ```
   [app] Lista en http://localhost:8000
   ```

4. Abre **http://localhost:8000** en el navegador.

Deja la terminal abierta mientras usas la interfaz. Para cerrarla, regresa a la terminal y presiona **Ctrl + C**.

> La página solo funciona desde **http://localhost:8000**. Si abres `index.html` con doble clic o con Live Server, verás el formulario, pero no se conectará con los modelos.

## 3. La pantalla

La pantalla tiene dos partes:

- **Panel izquierdo, "Sesión":** aquí eliges el modo, la vaca, los datos del ordeño y los modelos que quieres correr.
- **Panel derecho:** tiene cinco pestañas.
  - **Resultados:** las probabilidades, las alertas y la variable más importante de cada modelo.
  - **Gráficas:** los resultados de XGBoost en velocímetros, para ver de un vistazo si hay que revisar a la vaca (ver abajo).
  - **Variables de la sesión:** todos los valores con los que se hizo la predicción.
  - **Desempeño de los modelos:** qué tan bien funciona cada modelo en los datos de prueba.
  - **Glosario:** qué significa cada variable que puede aparecer como la más importante y qué revisar cuando aparece.

### Pestaña Gráficas

Muestra los resultados de **XGBoost v5**, el modelo principal, en velocímetros. Para verla, XGBoost debe estar activado en **Modelos**.

- **Franja de arriba:** responde "¿hago algo con esta vaca?".
  - En rojo: "Revisar a la vaca 1203: alerta en falla mecánica".
  - En ámbar: "Vigilar a la vaca…", cuando algún evento está cerca del umbral.
  - En azul: "Sin alertas: ordeño normal…".
- **Alguna anomalía, en grande:**
  - un velocímetro con el número debajo;
  - debajo, una escala de las tres zonas con su color y su rango (**Normal**, **Vigilar** y **Alerta**). La zona donde cae la aguja aparece en negritas y con su ícono, y las otras atenuadas. Es solo una leyenda: no se puede presionar;
  - una frase que lo resume (por ejemplo, "Está cerca del umbral: 33.7 %; la alerta empieza en 33.8 %");
  - lo que más pesó y qué revisar.
- **Cada evento:** un velocímetro chico para falla mecánica, salud y conducta, con su estado, lo que más pesó y qué revisar.
- **Cómo leer un velocímetro:** va de 0 % a 100 %, y la aguja marca la probabilidad.

  | Zona | Color | Rango |
  |---|---|---|
  | Normal | Azul | Por debajo de la mitad del umbral |
  | Vigilar | Ámbar | Desde la mitad del umbral hasta el umbral |
  | Alerta | Rojo | Desde el umbral de alerta del modelo |

  La marca oscura sobre el arco es el umbral. Como los umbrales son bajos (de 10 % a 34 %), la zona roja ocupa casi todo el arco: la alerta empieza pronto.
- **Al pasar el cursor** (o tocar un velocímetro en el celular) se ven la probabilidad y el umbral con un decimal, el estado y la variable más importante.

El estado nunca depende solo del color: siempre lleva ícono y texto. Se usa azul y no verde para "Normal" porque el verde y el ámbar no se distinguen con daltonismo.

### Nombres de vacas y robots

Las vacas y los robots aparecen con su nombre real (el número de la vaca y **VMS 1**, **VMS 2** o **VMS 3**) si en la computadora está el archivo `Data/mapa_id_vacas.csv`, que se genera al anonimizar los datos (`src/scripts/anonymize_data.py`). Sin ese archivo aparecen los identificadores anonimizados (por ejemplo `cdee4688` o `7063e4b1`).

> El mapa permite saber qué vaca es cada identificador: nunca lo subas al repositorio (`Data/` está en `.gitignore`) ni compartas capturas de pantalla con números reales de vacas fuera del equipo y del socio.

Arriba del panel izquierdo hay dos modos:

| Modo | Para qué sirve |
|---|---|
| **Nueva sesión** | Simular un ordeño que no está en los datos: capturas lo que se sabe cuando la vaca entra al robot. |
| **Del dataset** | Predecir un ordeño que ya ocurrió y comparar la predicción con lo que realmente pasó. |

En los dos modos, la casilla **Modelos** te deja elegir cuáles correr. Por defecto están los tres.

## 4. Predecir una sesión nueva

1. Elige **Nueva sesión**.
2. Elige la **vaca**. La lista está ordenada por la fecha de su último ordeño, de la más reciente a la más antigua. Debajo aparece la fecha y hora de ese último ordeño.
3. Revisa los campos del ordeño. Vienen prellenados con valores del último ordeño de la vaca, así que puedes predecir sin cambiar nada o ajustar solo lo que quieras probar.
4. Presiona **Predecir**. Con los tres modelos tarda de 1 a 3 segundos.

### Campos

| Campo | Qué capturar | Ejemplo |
|---|---|---|
| **Fecha** y **Hora** | Cuándo entra la vaca al robot. Debe ser posterior al último ordeño de la vaca y no después de la **fecha máxima**, que aparece debajo del campo. | 28/01/2026, 20:00 |
| **Máquina** | El robot donde se ordeña: VMS 1, VMS 2 o VMS 3. Sin el mapa de identificadores, VMS 1 aparece como `referencia` (la máquina más frecuente del hato, que no tiene columna propia) y VMS 3 y VMS 2 como `7063e4b1` y `86696b37`. | VMS 1 |
| **Núm. de ordeño** | El número de ordeño del día para esa vaca (1, 2, 3…). | 2 |
| **Temperatura (°C)** | La temperatura a esa hora. | 14 |
| **Humedad (%)** | La humedad relativa, de 0 a 100. | 45 |
| **Horas con THI ≥ 68 (24 h previas)** | Cuántas de las 24 horas anteriores tuvieron estrés por calor (THI de 68 o más), de 0 a 24. En días frescos suele ser 0. | 0 |

Con la temperatura y la humedad, la página calcula y muestra el **THI** (índice de temperatura y humedad) de la sesión.

### Qué se calcula solo

Todo lo demás que usan los modelos sale del historial de la vaca y del hato, con las mismas funciones que se usaron para preparar los datos de entrenamiento. Por ejemplo:

- las horas desde el ordeño anterior;
- lo que pasó en los ordeños previos: eventos, producción, MDI, conductividad, sangre y patadas;
- las tasas de eventos de la vaca en 30 días y del hato en 7 días;
- el calendario: mes, hora, días en leche y días en el hato;
- para la LSTM, la secuencia de los 30 ordeños anteriores.

Puedes ver todos esos valores en la pestaña **Variables de la sesión**.

## 5. Predecir un ordeño del dataset

1. Elige **Del dataset**.
2. Elige la **vaca**.
3. En **Ordeño**, elige la fecha y hora. La lista empieza por el más reciente.
4. Presiona **Predecir**.

Este modo usa los datos reales de ese ordeño y su historial. Además de la predicción, cada evento muestra **Real: ocurrió** o **Real: no ocurrió**, así que puedes comparar lo que predijo cada modelo con lo que pasó.

> Para revisar el desempeño en ordeños que los modelos no vieron al entrenar, elige ordeños del periodo de prueba: del 05/10/2025 en adelante.

## 6. Cómo leer los resultados

### Tabla comparativa

Arriba de la pestaña **Resultados** hay una tabla con una fila por evento y una columna por modelo. Cada celda tiene la probabilidad y la etiqueta **alerta** cuando la probabilidad llega al umbral de ese modelo. En el modo **Del dataset** se agrega la columna **Real**.

### Tarjetas por modelo

Debajo, cada modelo tiene una tarjeta por evento con:

| Elemento | Significado |
|---|---|
| **Porcentaje grande** | Probabilidad calibrada de que el evento ocurra en este ordeño. "Calibrada" significa que se ajustó para que, por ejemplo, de los ordeños con 30 % ocurra el evento en aproximadamente 3 de cada 10. |
| **Barra** | La parte de color es la probabilidad; la línea vertical marca el **umbral de alerta**. |
| **Alerta / Sin alerta** | Hay alerta cuando la probabilidad llega al umbral. Cada modelo y evento tiene su propio umbral, elegido para detectar al menos el 80 % de los eventos en validación. Por eso un 15 % puede ser alerta en falla mecánica y un 30 % no serlo en alguna anomalía. |
| **Real** | Solo en el modo **Del dataset**: si el evento ocurrió. |
| **Variable más importante** | La variable que más aumentó el riesgo en este ordeño (ver abajo). |

### La variable más importante

Cada tarjeta muestra solo la variable que más aumentó el riesgo, que es la que mejor explica la predicción del modelo. Muestra:

- **El nombre de la variable en palabras.** Por ejemplo "MDI del ordeño anterior" o "Patada en el ordeño anterior".
- **Para la LSTM, de qué ordeño viene:** "ordeño anterior", "hace 3 ordeños", etc. Si no dice nada, la variable es del ordeño actual.
- **El valor en sus unidades:** horas, días, kg, °C, porcentajes, "sí" o "no", el nombre del robot. El MDI se muestra en su escala original (los datos lo guardan en logaritmo) junto con si está en alerta (desde 1.4).
- **▲ aumenta el riesgo.**
- **¿Qué significa y qué hacer?:** al abrirlo explica la variable sin tecnicismos y sugiere qué revisar. Abajo, en gris, vienen el nombre técnico de la variable, su valor tal como lo recibe el modelo y su aporte en log-odds (mientras más grande, más pesó).

Las acciones son sugerencias de qué revisar, no un diagnóstico: confirma con el veterinario o el técnico. Las variables del hato, del calendario o del robot (por ejemplo "Vacas ordeñadas en la semana" o "Hora del ordeño") no dicen nada de la vaca en sí; su acción lo indica.

La variable se compara contra una "vaca promedio" del entrenamiento. Si ninguna variable sube el riesgo, la tarjeta dice "Ninguna variable aumenta el riesgo frente a una vaca promedio".

> **La variable más importante es una asociación, no una causa.** Indica qué pesó más en la predicción del modelo, no qué provocó el evento.

## 7. Variables de la sesión y desempeño de los modelos

### Variables de la sesión

Esta pestaña muestra los valores con los que se predijo el último ordeño, con el nombre en palabras, el nombre técnico debajo y el valor en sus unidades. Sirve para revisar qué recibieron los modelos, sobre todo en una sesión nueva. Casi todas describen ordeños anteriores de la vaca o del hato, no el ordeño actual. **Sin dato** significa que no había dato en el historial.

### Glosario

Una tarjeta por variable con qué significa y qué revisar. Es la misma explicación que aparece en "¿Qué significa y qué hacer?" en cada tarjeta.

### Desempeño de los modelos

Muestra, para cada modelo y evento, sus resultados en los datos de prueba (el 15 % más reciente de los ordeños):

| Métrica | Qué significa |
|---|---|
| **F1** | Balance entre detectar eventos y no dar falsas alarmas. Va de 0 a 1; más alto es mejor. |
| **Recall** | De los eventos que ocurrieron, qué proporción detectó el modelo. |
| **Precisión** | De las alertas que dio el modelo, qué proporción fueron eventos reales. |
| **Brier** | Qué tan bien calibradas están las probabilidades. Más bajo es mejor. |

Al final está la **línea base de persistencia**, una regla sin modelo: alertar si el ordeño anterior de la vaca tuvo el evento. Sirve como referencia; un modelo útil debe superarla.

## 8. Avisos y mensajes de error

### Avisos

Aparecen en un recuadro arriba de los resultados. La predicción se hace igual, pero hay que leerla con cuidado:

| Aviso | Qué significa |
|---|---|
| *Pasaron más de 48 h desde el último ordeño: la sesión se marca como regreso tras periodo seco…* | La vaca lleva mucho sin ordeñarse. Es una situación poco común en los datos y pesa mucho en las predicciones. |
| *Pasaron 45 días o más desde la última visita: empieza un bloque de historial nuevo…* | La vaca se trata como si regresara después de una ausencia larga: no hay ordeño previo, así que la LSTM no tiene secuencia y los demás modelos tienen menos información. |

### Errores

| Mensaje | Qué hacer |
|---|---|
| *La fecha debe ser a más tardar el …* | La fecha está después del fin del dataset más un día. Las tasas del hato usan los 7 días anteriores y no hay datos después de esa fecha. Elige una fecha anterior. |
| *La vaca no tiene ordeños anteriores a esa fecha.* | La fecha es anterior al primer ordeño de la vaca. Elige una fecha posterior. |
| *Faltan datos de la captura.* | Hay un campo vacío. Complétalos todos. |
| *La temperatura, la humedad, las horas y el número de ordeño deben ser números.* | Algún campo tiene texto. Escribe solo números. |
| *Elige al menos un modelo.* | Marca al menos una casilla en **Modelos**. |
| *No se pudo conectar con el servidor…* | El servidor no está corriendo o abriste la página fuera de http://localhost:8000. Ver [Problemas comunes](#10-problemas-comunes). |

## 9. Limitaciones

- **Fechas:** la sesión nueva solo puede ser, como máximo, el día siguiente al último dato del dataset (con el dataset actual, el 29/01/2026). Para fechas posteriores hacen falta datos nuevos.
- **Datos de prueba, no producción:** la interfaz usa los modelos del proyecto con los datos actuales. No es el sistema final de despliegue ni usa un modelo reentrenado con todos los datos.
- **Sesiones capturadas:** no se guardan. Al recargar la página se pierden.
- **Vacas sin historial suficiente:** con pocos ordeños previos, la LSTM tiene menos información y las predicciones son menos confiables.
- **Uso local:** la interfaz corre solo en la computadora donde se inicia y no es accesible desde otras computadoras.

## 10. Problemas comunes

| Problema | Solución |
|---|---|
| La página no carga en http://localhost:8000. | Revisa que la terminal muestre `[app] Lista en http://localhost:8000`. Si se cerró, vuelve a correr `python3 src/INTERFAZ/app.py`. |
| La página se ve sin estilos o no responde. | La abriste con doble clic o Live Server. Ábrela en http://localhost:8000. Si ya estás ahí, recarga sin caché con **Cmd + Shift + R** (Mac) o **Ctrl + Shift + R** (Windows). |
| Al iniciar: `XGBoost Library (libxgboost.dylib) could not be loaded`. | Falta OpenMP en Mac: `brew install libomp`. |
| Al iniciar: `No such file or directory: ... dataset_vacas_integrado.csv`. | Falta el dataset. Genéralo con los pasos del README. |
| Al iniciar: un error de columnas que faltan (`KeyError`). | El dataset no tiene las variables de historial. Corre `python3 src/MODELING/LSTM/build_history_features.py`. |
| Al iniciar: `Address already in use` o el puerto 8000 está ocupado. | Ya hay otra copia de la interfaz corriendo. Ciérrala con **Ctrl + C** en su terminal, o usa esa misma. |
| TensorFlow se cierra al iniciar en Mac. | Puede ser un choque con `pyarrow`. Si no lo necesitas para otro proyecto, desinstálalo con `python3 -m pip uninstall pyarrow`. |
