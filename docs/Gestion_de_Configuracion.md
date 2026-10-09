# Gestión de la Configuración

## 1. Propósito y alcance

Establecer los lineamientos para identificar, controlar, versionar y proteger los elementos de configuración del proyecto, con el fin de asegurar la integridad, la trazabilidad y la consistencia de los productos de trabajo (código y documentación).

Aplica a todos los miembros del equipo. Cualquier elemento creado o modificado a partir de la fecha de implementación de este documento debe cumplir con lo aquí definido.

## 2. Terminología

| Término | Definición |
|---|---|
| **CI** (Elemento de Configuración) | Producto de trabajo cuya alteración puede afectar la calidad, la trazabilidad, la entrega o el cumplimiento del proyecto, y que por ello se identifica y se controla. |
| **Línea base** | Versión aprobada y congelada de un conjunto de CIs, que sirve de referencia y solo puede modificarse mediante un cambio controlado. |
| **Repositorio** | Sistema oficial donde reside un CI. Para este proyecto, GitHub. |
| **Release / Tag** | Marca inmutable en GitHub que identifica una línea base. |

## 3. Política

1. Todo producto de trabajo relevante debe identificarse como CI cuando su alteración pueda afectar la calidad, la trazabilidad, la entrega o el cumplimiento del proyecto.
2. Los CIs se clasifican por tipo: código, documento, dependencias/configuración, datos de referencia.
3. Todo cambio a un CI debe hacerse mediante el flujo de control de cambios (sección 7). No se aceptan modificaciones directas a `main` ni a `develop`.
4. Los CIs se mantienen bajo control de versiones en Git. El historial de commits y de Pull Requests es el registro de quién cambió qué y cuándo.
5. Las líneas base se establecen en puntos de control definidos: entrega al socio formador y aprobación formal de entregables.
6. Nunca se versionan secretos ni credenciales (`.env`, llaves `*.pem`, tokens). Tampoco se versionan datos sensibles ni crudos de la granja; los datos se mantienen fuera del repositorio.

## 4. Identificación de CIs

| Categoría | CI | Ubicación |
|---|---|---|
| Código fuente | Scripts de análisis y procesamiento de datos | `src/` |
| Análisis | Análisis descriptivo y EDA | `analisis_basico_descriptivo_datos_vacas/`, `src/` |
| Documentación de entendimiento del negocio | Reporte de Business Understanding, One Page, Memorándum de entendimiento, WBS | `docs/BU/` |
| Documentación de entendimiento de los datos | Perfil y descripción de datos, figuras | `docs/DU/` |
| Dependencias | Archivo de requerimientos | `requirements.txt` |
| Documentación general | README, licencia | `README.md`, `LICENSE` |
| Gestión de la configuración | Este documento | `docs/Gestion_de_Configuracion.md` |

No se consideran CIs: archivos temporales del sistema (`.DS_Store`), entornos virtuales (`venv/`), cachés y salidas intermedias reproducibles.

## 5. Convenciones de nomenclatura

### 5.1 Ramas

| Rama | Uso | Reglas |
|---|---|---|
| `main` | Versión estable y entregas al socio formador. | Protegida. Solo recibe merges desde `release/*` (o desde `develop` al cerrar una entrega). Nunca commit directo. |
| `develop` | Integración del trabajo en curso. | Protegida. Recibe PRs de ramas de trabajo. Nunca commit directo. |
| `feature/<descripcion>` | Nueva funcionalidad, análisis o documento. | Se crea desde `develop`. |
| `fix/<descripcion>` | Corrección de un error. | Se crea desde `develop` (o desde `main` si es urgente, y luego se integra a `develop`). |
| `docs/<descripcion>` | Cambios únicamente de documentación. | Se crea desde `develop`. |
| `release/vX.Y.Z` | Preparación y congelamiento de una línea base. | Se crea desde `develop`; solo admite correcciones menores. |

Reglas de nombre:

- Minúsculas, palabras separadas por guion bajo (`_`), sin tildes ni espacios.
- Descriptivo y corto; debe indicar qué se hace, no quién lo hace.
- Una rama por cambio; se elimina después del merge.

Ejemplos: `feature/gestion_configuracion`, `feature/descripcion_datos`, `fix/path_carpeta_raiz`, `docs/reporte_bu`, `release/v1.0.0`.

### 5.2 Commits

Formato:

```
<tipo>: <descripción breve en español>
```

| Tipo | Cuándo usarlo |
|---|---|
| `feature` | Se agrega una funcionalidad, un script o un documento nuevo. |
| `fix` | Se corrige un error. |
| `update` | Se modifica algo existente sin ser corrección (renombrar, ajustar, mejorar). |
| `docs` | Cambios menores solo de documentación. |

Reglas:

- Una línea de máximo ~72 caracteres, en español, en minúscula tras los dos puntos y sin punto final.
- Un commit = un cambio lógico. No mezclar temas distintos.
- Commits y PRs deben estar en el mismo idioma (español).
- Prohibido `--force`, `reset --hard` o `--no-verify` sobre ramas compartidas sin acuerdo explícito del equipo.
- Nunca incluir secretos, credenciales ni datos sensibles.

Ejemplos válidos:

```
feature: agrega archivo de requerimientos
fix: corrige el path de la carpeta raiz
update: cambiar el nombre del archivo eda
fix: borrar la carpeta de Data por seguridad
```

### 5.3 Pull Requests

- Todo cambio llega a `develop` (o a `main` desde `release/*`) mediante un Pull Request. No se hace push directo.
- **Título:** mismo formato que un commit (`feature: ...`, `fix: ...`).
- **Plantilla:** todo PR debe usar la plantilla del repositorio (`.github/pull_request_template.md`), que GitHub carga automáticamente, y completar todas sus secciones:
  - **Resumen:** el problema y la solución propuesta.
  - **Tipo de cambio:** documentación (`docs/`), código fuente (`src/`) u otro.
  - **Cambios realizados:** lista de los cambios incluidos.
  - **Lista de verificación:** se completa la parte que corresponde al tipo de cambio.
    - Documentación: la información es correcta, clara y actualizada; se revisaron ortografía, enlaces y rutas; los diagramas, imágenes o adjuntos funcionan.
    - Código: sigue las convenciones del proyecto; se probó localmente; se agregaron o actualizaron pruebas cuando corresponde; se actualizó la documentación relacionada cuando corresponde.
  - **Evidencia:** capturas, resultados de pruebas o enlaces relevantes, si aplica.
  - **Notas para revisión:** decisiones, limitaciones o aspectos que requieran atención especial.
- **Issue:** si se requiere formalizar la petición antes de implementar, se abre primero un issue y se vincula al PR.
- **Revisor:** una persona distinta del autor. Al aprobar y hacer el merge asume la responsabilidad de que el cambio cumple los estándares, incluidos los de este documento (nomenclatura de ramas y commits, y que no se incluyan secretos, datos crudos ni archivos del sistema).
- **Resultado:** el revisor aprueba, o rechaza con *Request changes* indicando qué corregir. Un PR no se aprueba con la lista de verificación incompleta.
- **Merge:** se hace una vez aprobado.

### 5.4 Versionado y tags

Se usa versionado semántico `vMAJOR.MINOR.PATCH`:

| Componente | Se incrementa cuando… |
|---|---|
| `MAJOR` | Hay un cambio de entrega mayor (por ejemplo, un nuevo modelo o fase completa del proyecto). |
| `MINOR` | Se agrega funcionalidad o documentación nueva compatible. |
| `PATCH` | Se hacen correcciones menores. |

Ejemplos: `v0.1.0` (primera línea base), `v0.2.0` (nueva funcionalidad o documento aprobado), `v1.0.0` (entrega al socio formador).

Los documentos de `docs/` que lo requieran pueden llevar además una tabla de control de versiones propia (versión, descripción, autor, fecha).

## 6. Líneas base y releases

### 6.1 Cuándo crear una línea base

- Al realizar una entrega al socio formador.
- Al aprobar formalmente un entregable (por ejemplo, un reporte o un conjunto de scripts de análisis).

### 6.2 Cómo crear un release en GitHub

1. Verificar que `develop` (o la rama `release/*`) esté al día y que todos los cambios incluidos hayan sido aprobados mediante PR.
2. Mezclar a `main` mediante PR aprobado.
3. En GitHub: **Releases → Draft a new release**.
4. Asignar el tag `vX.Y.Z` según la sección 5.4.
5. Documentar en la descripción los cambios principales y los PRs incluidos.
6. Publicar el release.

Una línea base solo puede asociarse a cambios aprobados: no se establece línea base de algo que está en revisión.

## 7. Control de cambios

Flujo estándar:

```
issue (opcional) → rama desde develop → commits → Pull Request → revisión → merge
```

1. **Sincronizar:** antes de empezar, `git fetch` y verificar que la rama local esté al día con su upstream y con `develop`.
2. **Crear la rama** con la nomenclatura de la sección 5.1.
3. **Trabajar** con commits pequeños y con el formato de la sección 5.2.
4. **Abrir el PR** hacia `develop` con descripción y referencia al requisito o issue.
5. **Revisión** por una persona distinta del autor, con la checklist de la sección 5.3.
6. **Merge** una vez aprobado; la rama de trabajo se elimina.

Trazabilidad: cualquier cambio en un requisito o en el alcance debe reflejarse en la documentación correspondiente de `docs/` en el mismo PR.

## 8. Rollback (retrocesos)

Cuando un CI o un release genere un problema funcional, documental o de calidad:

- El retroceso se hace únicamente hacia la **última versión aprobada y liberada** anterior, registrada como release o tag.
- Deben cumplirse los criterios: existe una versión anterior aprobada y el rollback queda documentado como un nuevo cambio controlado.
- No se regresa a un commit arbitrario sin trazabilidad ni aprobación formal.

Procedimiento:

1. Identificar el tag o release de destino.
2. Crear una rama `fix/rollback_<descripcion>` desde `develop`.
3. Revertir con `git revert` (conserva el historial) en lugar de reescribirlo.
4. Abrir un PR explicando el motivo y la versión de destino, y seguir el flujo normal de revisión.

Está prohibido reescribir el historial compartido (`reset --hard`, `push --force`) para realizar un rollback.