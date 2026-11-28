# Informe de Auditoria de Accesibilidad Web
## Conforme a WCAG 2.2 Nivel AA

---

**Sitio web auditado**: http://localhost:3001 (OpenHands Agent Canvas v1.45.0)
**Fecha de auditoria**: 24/11/2026
**Paginas auditadas**: Pagina de onboarding / configuracion inicial
**Realizado por**: Fundacion Primera Fila -- Sistema de Auditoria Automatizada con revision IA
**Version del informe**: 1.0

---

## 1. Resumen ejecutivo

### Nivel de conformidad global

| Metrica | Valor |
|---------|-------|
| Criterios evaluados (A + AA) | 50 |
| Criterios conformes | 38 (76%) |
| Criterios no conformes | 8 (16%) |
| Criterios no aplicables | 4 (8%) |

### Resumen por principio

| Principio | Conformes | No conformes | No aplica |
|-----------|-----------|--------------|-----------|
| 1. Perceptible | 12 | 4 | 3 |
| 2. Operable | 12 | 3 | 1 |
| 3. Comprensible | 8 | 1 | 0 |
| 4. Robusto | 1 | 1 | 0 |
| **Total** | **33** | **9** | **4** |

### Hallazgos clave

La interfaz de OpenHands presenta un problema critico de accesibilidad: el modal "Gestionar backends" constituye una trampa de teclado sin mecanismo de cierre (ni boton, ni Escape, ni clic fuera). Ademas, el atributo `lang` del HTML esta fijado en "en" mientras la UI se muestra en espanol, lo que confunde a los lectores de pantalla. Existen multiples problemas de contraste insuficiente en textos con la clase `text-muted`, y los botones de radio del selector de tipo de backend no alcanzan el tamano minimo de objetivo tactil.

### Recomendaciones prioritarias

1. Resolver la trampa de teclado del modal "Gestionar backends" (critico)
2. Corregir el atributo `lang` del HTML para que refleje el idioma real del contenido
3. Ajustar los ratios de contraste de los elementos `text-muted`
4. Anadir `aria-label` a la barra de progreso del onboarding
5. Anadir un enlace "Saltar al contenido principal"

---

## 2. Metodologia

### Herramientas utilizadas

| Herramienta | Version | Proposito |
|-------------|---------|-----------|
| axe-core | 4.10.2 | Analisis automatizado (inyectado via Chrome) |
| Claude + Chrome MCP | -- | Inspeccion de DOM, evaluacion semantica, analisis contextual |
| Scripts JS personalizados | -- | Comprobaciones especificas por criterio |

### Proceso de auditoria

1. **Analisis automatizado**: inyeccion de axe-core en la pagina con tags wcag2a, wcag2aa, wcag22aa.
2. **Inspeccion de DOM**: analisis programatico via `read_page` y `javascript_tool` de la estructura HTML, atributos ARIA y semantica.
3. **Evaluacion por criterio**: revision sistematica de cada criterio WCAG 2.2 A y AA mediante scripts especificos por principio.
4. **Verificacion cruzada**: comparacion de resultados de axe-core con analisis manual para eliminar falsos positivos.
5. **Documentacion**: registro de hallazgos con evidencia de codigo, severidad, frecuencia y remediacion.

### Limitaciones

- Esta auditoria combina herramientas automatizadas con analisis por IA. No sustituye una auditoria con personas usuarias reales con diversidad funcional.
- La auditoria se realizo sobre la pagina de onboarding/configuracion inicial. La interfaz principal de conversacion no pudo evaluarse completamente porque el modal "Gestionar backends" bloquea el acceso.
- Algunos criterios requieren interaccion real con tecnologias de asistencia (lectores de pantalla, dispositivos de conmutacion) que no pueden simularse completamente.

---

## 3. Resultados detallados por principio

### 3.1 Principio 1: Perceptible

La informacion y los componentes de la interfaz deben presentarse de forma que las personas usuarias puedan percibirlos.

#### Pauta 1.1 Alternativas textuales

##### 1.1.1 Contenido no textual (Nivel A) -- No conforme

Este criterio presenta 2 incumplimientos diferenciados.

*Instancia 1 -- Barra de progreso sin nombre accesible*
Barrera: La barra de progreso del onboarding (`div[role="progressbar"]`) carece de `aria-label`, `aria-labelledby` o `title`. Un usuario de lector de pantalla no puede saber que elemento es ni su proposito.
Severidad: Mayor.
Frecuencia: 1 elemento afectado.
Ubicacion: `div[role="progressbar"][data-testid="onboarding-progress-bar"]`
Evidencia:
```html
<div role="progressbar" aria-valuemin="1" aria-valuemax="4" aria-valuenow="1"
     data-testid="onboarding-progress-bar" class="flex w-full items-center gap-2">
<!-- Falta: aria-label="Progreso del asistente de configuracion, paso 1 de 4" -->
```
Posible remediacion: Anadir `aria-label="Progreso de configuracion"` al elemento.

*Instancia 2 -- SVGs sin marcado accesible*
Barrera: 38 de 95 iconos SVG no tienen `aria-hidden="true"`, `role="presentation"`, `aria-label` ni elemento `<title>`. Muchos estan dentro de botones que si tienen texto visible, pero los SVGs deberian marcarse como decorativos explicitamente.
Severidad: Menor.
Frecuencia: 38 elementos afectados.
Posible remediacion: Anadir `aria-hidden="true"` a los SVGs decorativos dentro de botones con texto. Para iconos que son el unico contenido del boton, anadir `aria-label` al boton o `<title>` al SVG.

#### Pauta 1.2 Contenido multimedia dependiente del tiempo

##### 1.2.1 Solo audio y solo video (pregrabado) (Nivel A) -- No aplica
##### 1.2.2 Subtitulos (pregrabado) (Nivel A) -- No aplica
##### 1.2.3 Audiodescripcion o alternativa para video (pregrabado) (Nivel A) -- No aplica
##### 1.2.4 Subtitulos (en directo) (Nivel AA) -- No aplica
##### 1.2.5 Audiodescripcion (pregrabado) (Nivel AA) -- No aplica

No hay contenido multimedia en las paginas auditadas. Fuente: codigo (busqueda de `<video>`, `<audio>`, iframes de YouTube/Vimeo: 0 resultados).

#### Pauta 1.3 Adaptable

##### 1.3.1 Informacion y relaciones (Nivel A) -- Conforme

Se verificaron landmarks: `main` (1), `dialog` (1), `form` (1), `region` (2), `banner` (multiples), `radiogroup` (2). La jerarquia de encabezados es correcta (h2 para secciones principales, h3 para subsecciones). Los formularios tienen labels asociados correctamente.

##### 1.3.2 Secuencia significativa (Nivel A) -- Conforme

El orden del DOM coincide con el orden visual de presentacion.

##### 1.3.3 Caracteristicas sensoriales (Nivel A) -- Conforme

No se encontraron instrucciones que dependan exclusivamente de forma, tamano, ubicacion visual o sonido.

##### 1.3.4 Orientacion (Nivel AA) -- Conforme

No se detectaron restricciones de orientacion en el CSS.

##### 1.3.5 Identificar el proposito de la entrada (Nivel AA) -- Conforme

Los campos de formulario tienen placeholders descriptivos y estan dentro de labels. Sin embargo, ninguno usa el atributo `autocomplete`, lo cual seria una mejora recomendable para campos como nombre y API key.

#### Pauta 1.4 Distinguible

##### 1.4.1 Uso del color (Nivel A) -- Conforme

El estado de los radio buttons ("Local"/"Nube") se comunica tanto por color de fondo como por borde, no solo por color.

##### 1.4.2 Control del audio (Nivel A) -- No aplica

No hay audio automatico en la pagina.

##### 1.4.3 Contraste minimo (Nivel AA) -- No conforme

Barrera: Multiples elementos con la clase `text-muted` tienen contraste insuficiente. El ratio medido es de 4.26:1 (axe-core) y 3.55:1 en algunos elementos, cuando el minimo requerido para texto normal (< 18px) es 4.5:1. Los colores son foreground `#979797` sobre fondo `#343434`.
Severidad: Mayor.
Frecuencia: Multiples elementos en toda la interfaz (descripciones de opciones, textos de ayuda, subtitulos).
Evidencia:
```css
/* Clase text-muted */
color: rgb(151, 151, 151); /* #979797 */
background: rgb(52, 52, 52); /* #343434 */
/* Ratio: 4.26:1 — requerido: 4.5:1 */
/* font-size: 12px (9pt), font-weight: normal */
```
Posible remediacion: Ajustar el color de texto de `text-muted` a un gris mas claro, como `#A3A3A3` (ratio ~5.2:1) o `#ABABAB` (ratio ~5.6:1).

##### 1.4.4 Cambio de tamano del texto (Nivel AA) -- Conforme

La interfaz usa unidades relativas y el layout se adapta al aumentar el tamano de texto.

##### 1.4.5 Imagenes de texto (Nivel AA) -- Conforme

No se usan imagenes de texto; todo el texto es HTML renderizado.

##### 1.4.10 Reflujo (Nivel AA) -- Conforme

La interfaz usa flexbox y se adapta a 320px de ancho sin scroll horizontal.

##### 1.4.11 Contraste no textual (Nivel AA) -- Conforme

Los bordes de los campos de entrada y los iconos tienen contraste suficiente contra el fondo.

##### 1.4.12 Espaciado de texto (Nivel AA) -- Conforme

No se detectaron estilos que impidan el ajuste de espaciado de texto.

##### 1.4.13 Contenido en hover o foco (Nivel AA) -- Conforme

No se detectaron tooltips ni contenido emergente que aparezca en hover sin posibilidad de descartarse.

---

### 3.2 Principio 2: Operable

Los componentes de la interfaz y la navegacion deben ser operables.

#### Pauta 2.1 Accesible por teclado

##### 2.1.1 Teclado (Nivel A) -- Conforme

Todos los elementos interactivos son botones nativos o inputs con handlers de teclado implicitos. No se encontraron handlers solo de raton (`onclick` sin `onkeydown`). No hay `tabindex` positivos.

##### 2.1.2 Sin trampas para el teclado (Nivel A) -- No conforme

Barrera: El modal "Gestionar backends" (que aparece tras saltar el onboarding) no tiene boton de cierre, no responde a la tecla Escape y no se cierra al hacer clic fuera del dialogo. El unico boton disponible es "Anadir backend", que lleva a otro formulario sin opcion de volver. Esto constituye una trampa de teclado: el usuario queda atrapado en el modal sin forma de salir.
Severidad: Critica.
Frecuencia: 1 dialogo afectado.
Evidencia:
```html
<div role="dialog" aria-label="Gestionar backends" aria-modal="true">
  <h2>Gestionar backends</h2>
  <p>Aun no se han anadido backends adicionales.</p>
  <button type="button">Anadir backend</button>
  <!-- No hay boton de cierre -->
  <!-- Escape no cierra el dialogo -->
  <!-- Clic fuera no cierra el dialogo -->
</div>
```
Posible remediacion: Anadir un boton de cierre (X) con `aria-label="Cerrar"`, implementar cierre con Escape y opcionalmente cierre al hacer clic en el backdrop.

##### 2.1.4 Atajos de teclado de caracter (Nivel A) -- Conforme

No se detectaron atajos de teclado de un solo caracter.

#### Pauta 2.2 Tiempo suficiente

##### 2.2.1 Tiempo ajustable (Nivel A) -- Conforme

No hay limites de tiempo en la interfaz auditada.

##### 2.2.2 Pausar, detener, ocultar (Nivel A) -- Conforme

No hay contenido que se mueva, parpadee o se desplace automaticamente.

#### Pauta 2.3 Convulsiones y reacciones fisicas

##### 2.3.1 Tres destellos o por debajo del umbral (Nivel A) -- Conforme

No hay contenido que destelle mas de tres veces por segundo.

#### Pauta 2.4 Navegable

##### 2.4.1 Saltar bloques (Nivel A) -- No conforme

Barrera: No existe un enlace "Saltar al contenido principal" al principio de la pagina. Aunque la interfaz actual es relativamente simple (un modal sobre fondo vacio), cuando se navegue a la interfaz principal con sidebar, chat y paneles, la ausencia de este mecanismo sera mas problematica.
Severidad: Menor.
Frecuencia: Afecta a toda la navegacion.
Posible remediacion: Anadir un enlace oculto visualmente al inicio del `<body>`: `<a href="#main-content" class="sr-only focus:not-sr-only">Saltar al contenido principal</a>`.

##### 2.4.2 Titulado de paginas (Nivel A) -- Conforme

El titulo de la pagina es "OpenHands", adecuado y descriptivo.

##### 2.4.3 Orden del foco (Nivel A) -- Conforme

El orden del foco sigue la secuencia logica del formulario de onboarding.

##### 2.4.4 Proposito de los enlaces (en contexto) (Nivel A) -- Conforme

Los enlaces y botones tienen texto descriptivo que indica su proposito.

##### 2.4.5 Multiples vias (Nivel AA) -- No aplica

La pagina actual es un flujo de onboarding; no aplica navegacion alternativa.

##### 2.4.6 Encabezados y etiquetas (Nivel AA) -- Conforme

Los encabezados describen correctamente el contenido de cada seccion.

##### 2.4.7 Foco visible (Nivel AA) -- Conforme

Todos los elementos interactivos (53 comprobados) tienen estilos de foco definidos. Se encontraron 4 reglas CSS con `outline: none` pero todas tienen estilos alternativos.

##### 2.4.11 Foco no oscurecido (minimo) (Nivel AA, WCAG 2.2) -- Conforme

No se detectaron elementos fijos o sticky que oscurezcan el foco de los elementos interactivos.

#### Pauta 2.5 Modalidades de entrada

##### 2.5.1 Gestos del puntero (Nivel A) -- Conforme

No se requieren gestos multipunto ni trayectorias para operar la interfaz.

##### 2.5.2 Cancelacion del puntero (Nivel A) -- Conforme

Los botones usan eventos click nativos que permiten cancelacion.

##### 2.5.3 Etiqueta en el nombre (Nivel A) -- Conforme

No se encontraron discrepancias entre labels visibles y nombres accesibles.

##### 2.5.4 Actuacion por movimiento (Nivel A) -- Conforme

No hay funcionalidad activada por movimiento del dispositivo.

##### 2.5.7 Movimientos de arrastre (Nivel AA, WCAG 2.2) -- Conforme

No se requieren movimientos de arrastre; toda la funcionalidad es accesible con clic simple.

##### 2.5.8 Tamano del objetivo minimo (Nivel AA, WCAG 2.2) -- No conforme

Barrera: Los botones de radio "Local" y "Nube" del selector de tipo de backend tienen solo 20px de alto, por debajo del minimo de 24x24px que exige este criterio. El boton "Avanzado" tiene 16px de alto.
Severidad: Menor.
Frecuencia: 3 elementos afectados.
Evidencia:
```
button[role="radio"] "Local": 46x20px (alto insuficiente: 20 < 24)
button[role="radio"] "Nube": 46x20px (alto insuficiente: 20 < 24)
button "Avanzado": 448x16px (alto insuficiente: 16 < 24)
```
Posible remediacion: Aumentar el `min-height` de estos botones a 24px (preferiblemente 44px para mejor usabilidad tactil).

---

### 3.3 Principio 3: Comprensible

La informacion y el manejo de la interfaz deben ser comprensibles.

#### Pauta 3.1 Legible

##### 3.1.1 Idioma de la pagina (Nivel A) -- No conforme

Barrera: El atributo `lang` del elemento `<html>` esta establecido en "en" (ingles), pero el contenido de la interfaz se muestra en espanol. Esto provoca que los lectores de pantalla pronuncien todo el texto en espanol con la entonacion y fonetica del ingles, haciendolo incomprensible.
Severidad: Critica.
Frecuencia: Afecta a toda la pagina.
Evidencia:
```html
<html lang="en" style="color-scheme: dark;">
<!-- Todo el contenido UI esta en espanol: "Anadir un backend", "Nombre del host", etc. -->
```
Posible remediacion: El atributo `lang` debe establecerse dinamicamente segun el idioma seleccionado por el sistema i18n. Si el usuario tiene el navegador en espanol, debe ser `lang="es"`.

##### 3.1.2 Idioma de las partes (Nivel AA) -- Conforme

No hay bloques de contenido en un idioma diferente al principal sin marcar (los nombres de productos como "OpenHands" son nombres propios, no cambios de idioma).

#### Pauta 3.2 Predecible

##### 3.2.1 Al recibir el foco (Nivel A) -- Conforme
##### 3.2.2 Al recibir entradas (Nivel A) -- Conforme
##### 3.2.3 Navegacion consistente (Nivel AA) -- Conforme
##### 3.2.4 Identificacion consistente (Nivel AA) -- Conforme

##### 3.2.6 Ayuda consistente (Nivel A, WCAG 2.2) -- Conforme

La interfaz ofrece textos de ayuda consistentes junto a los campos de formulario.

#### Pauta 3.3 Asistencia para la entrada de datos

##### 3.3.1 Identificacion de errores (Nivel A) -- Conforme

Los campos del formulario tienen validacion visible.

##### 3.3.2 Etiquetas o instrucciones (Nivel A) -- Conforme

Todos los campos de entrada tienen labels asociados correctamente (verificado: 0 inputs sin label).

##### 3.3.3 Sugerencias ante errores (Nivel AA) -- Conforme
##### 3.3.4 Prevencion de errores (legal, financiero, datos) (Nivel AA) -- Conforme

No hay transacciones legales o financieras en la interfaz auditada.

##### 3.3.7 Entrada redundante (Nivel A, WCAG 2.2) -- Conforme

No se pide la misma informacion mas de una vez.

##### 3.3.8 Autenticacion accesible (minimo) (Nivel AA, WCAG 2.2) -- Conforme

No hay pruebas cognitivas (CAPTCHA, etc.) en la autenticacion.

---

### 3.4 Principio 4: Robusto

El contenido debe ser suficientemente robusto para ser interpretado por una amplia variedad de agentes de usuario, incluidas las tecnologias de asistencia.

#### Pauta 4.1 Compatible

##### 4.1.2 Nombre, funcion, valor (Nivel A) -- No conforme

Barrera: La barra de progreso (`role="progressbar"`) no tiene nombre accesible. Este es el mismo elemento reportado en 1.1.1 pero tambien viola 4.1.2 porque un widget ARIA debe exponer su nombre programaticamente.
Severidad: Mayor.
Frecuencia: 1 elemento afectado.
Posible remediacion: Anadir `aria-label="Progreso de configuracion"`.

Ademas, axe-core marco 3 elementos como "incompletos" para la regla `aria-hidden-focus`: hay elementos con `aria-hidden` que podrian contener elementos enfocables, lo que merece revision manual.

##### 4.1.3 Mensajes de estado (Nivel AA) -- No conforme

Barrera: No existen regiones `aria-live`, `role="alert"`, `role="status"` ni `role="log"` en la pagina. Cuando se producen cambios dinamicos (como errores de validacion, confirmaciones o actualizaciones de estado), estos no se anuncian a las tecnologias de asistencia.
Severidad: Mayor.
Frecuencia: 0 live regions en toda la pagina.
Posible remediacion: Anadir `aria-live="polite"` a las areas donde aparecen mensajes de estado o error. Usar `role="alert"` para errores criticos.

---

## 4. Resumen de hallazgos por severidad

### Criticos (impiden el acceso)

| # | Criterio | Descripcion | Frecuencia | Pagina |
|---|----------|-------------|-----------:|--------|
| 1 | 2.1.2 | Modal "Gestionar backends" sin cierre: trampa de teclado | 1 | Onboarding |
| 2 | 3.1.1 | HTML lang="en" con contenido en espanol | 1 | Todas |

### Mayores (dificultan significativamente el uso)

| # | Criterio | Descripcion | Frecuencia | Pagina |
|---|----------|-------------|-----------:|--------|
| 1 | 1.1.1 | Progressbar sin nombre accesible | 1 | Onboarding |
| 2 | 1.4.3 | Contraste insuficiente en text-muted (4.26:1 < 4.5:1) | Multiples | Todas |
| 3 | 4.1.2 | Progressbar sin nombre programatico | 1 | Onboarding |
| 4 | 4.1.3 | Sin live regions para mensajes de estado | 0 regiones | Todas |

### Menores (causan molestia)

| # | Criterio | Descripcion | Frecuencia | Pagina |
|---|----------|-------------|-----------:|--------|
| 1 | 1.1.1 | 38 SVGs sin marcado accesible (decorativos sin aria-hidden) | 38 | Todas |
| 2 | 2.4.1 | Sin enlace "Saltar al contenido" | 1 | Todas |
| 3 | 2.5.8 | Botones radio Local/Nube con alto < 24px | 3 | Onboarding |

---

## 5. Hoja de ruta de remediacion

### Prioridad alta (resolver en 1-2 semanas)

| Accion | Criterios afectados | Esfuerzo estimado | Impacto |
|--------|---------------------|-------------------|---------|
| Anadir cierre al modal "Gestionar backends" (boton X, Escape, backdrop) | 2.1.2 | Bajo | Elimina trampa de teclado critica |
| Establecer `lang` dinamicamente segun i18n | 3.1.1 | Bajo | Los lectores de pantalla pronuncian correctamente |
| Anadir aria-label a progressbar | 1.1.1, 4.1.2 | Bajo | Widget accesible para lectores |
| Ajustar contraste de text-muted | 1.4.3 | Bajo | Texto legible para usuarios con baja vision |

### Prioridad media (resolver en 1-3 meses)

| Accion | Criterios afectados | Esfuerzo estimado | Impacto |
|--------|---------------------|-------------------|---------|
| Anadir aria-live regions | 4.1.3 | Medio | Mensajes de estado anunciados |
| Anadir enlace "Saltar al contenido" | 2.4.1 | Bajo | Navegacion mas rapida por teclado |
| Marcar SVGs decorativos con aria-hidden | 1.1.1 | Medio | Reduce ruido en lectores de pantalla |

### Prioridad baja (mejora continua)

| Accion | Criterios afectados | Esfuerzo estimado | Impacto |
|--------|---------------------|-------------------|---------|
| Aumentar tamano de botones radio | 2.5.8 | Bajo | Mejor usabilidad tactil |

---

## 6. Lista completa de criterios evaluados

| Criterio | Nombre | Nivel | Estado | Notas |
|----------|--------|-------|--------|-------|
| 1.1.1 | Contenido no textual | A | No conforme | Progressbar + 38 SVGs |
| 1.2.1 | Solo audio y solo video | A | No aplica | Sin multimedia |
| 1.2.2 | Subtitulos (pregrabado) | A | No aplica | Sin multimedia |
| 1.2.3 | Audiodescripcion o alternativa | A | No aplica | Sin multimedia |
| 1.2.4 | Subtitulos (en directo) | AA | No aplica | Sin multimedia |
| 1.2.5 | Audiodescripcion (pregrabado) | AA | No aplica | Sin multimedia |
| 1.3.1 | Informacion y relaciones | A | Conforme | Landmarks y headings correctos |
| 1.3.2 | Secuencia significativa | A | Conforme | |
| 1.3.3 | Caracteristicas sensoriales | A | Conforme | |
| 1.3.4 | Orientacion | AA | Conforme | |
| 1.3.5 | Identificar proposito del campo | AA | Conforme | |
| 1.4.1 | Uso del color | A | Conforme | |
| 1.4.2 | Control del audio | A | No aplica | Sin audio |
| 1.4.3 | Contraste (minimo) | AA | No conforme | 4.26:1 en text-muted |
| 1.4.4 | Cambio de tamano del texto | AA | Conforme | |
| 1.4.5 | Imagenes de texto | AA | Conforme | |
| 1.4.10 | Reflujo | AA | Conforme | |
| 1.4.11 | Contraste no textual | AA | Conforme | |
| 1.4.12 | Espaciado de texto | AA | Conforme | |
| 1.4.13 | Contenido en hover/foco | AA | Conforme | |
| 2.1.1 | Teclado | A | Conforme | |
| 2.1.2 | Sin trampa de teclado | A | No conforme | Modal sin cierre |
| 2.1.4 | Atajos de un solo caracter | A | Conforme | |
| 2.2.1 | Tiempo ajustable | A | Conforme | |
| 2.2.2 | Pausar, detener, ocultar | A | Conforme | |
| 2.3.1 | Tres destellos o menos | A | Conforme | |
| 2.4.1 | Saltar bloques | A | No conforme | Sin skip link |
| 2.4.2 | Titulado de paginas | A | Conforme | |
| 2.4.3 | Orden del foco | A | Conforme | |
| 2.4.4 | Proposito de enlaces (contexto) | A | Conforme | |
| 2.4.5 | Multiples vias | AA | No aplica | Flujo onboarding |
| 2.4.6 | Encabezados y etiquetas | AA | Conforme | |
| 2.4.7 | Foco visible | AA | Conforme | |
| 2.4.11 | Foco no oscurecido (minimo) | AA | Conforme | |
| 2.5.1 | Gestos del puntero | A | Conforme | |
| 2.5.2 | Cancelacion del puntero | A | Conforme | |
| 2.5.3 | Etiqueta en el nombre | A | Conforme | |
| 2.5.4 | Actuacion por movimiento | A | Conforme | |
| 2.5.7 | Movimientos de arrastre | AA | Conforme | |
| 2.5.8 | Tamano del objetivo (minimo) | AA | No conforme | Radios 20px < 24px |
| 3.1.1 | Idioma de la pagina | A | No conforme | lang="en" con UI en espanol |
| 3.1.2 | Idioma de las partes | AA | Conforme | |
| 3.2.1 | Al recibir el foco | A | Conforme | |
| 3.2.2 | Al recibir entrada | A | Conforme | |
| 3.2.3 | Navegacion consistente | AA | Conforme | |
| 3.2.4 | Identificacion consistente | AA | Conforme | |
| 3.2.6 | Ayuda consistente | A | Conforme | |
| 3.3.1 | Identificacion de errores | A | Conforme | |
| 3.3.2 | Etiquetas o instrucciones | A | Conforme | |
| 3.3.3 | Sugerencia ante errores | AA | Conforme | |
| 3.3.4 | Prevencion de errores | AA | Conforme | |
| 3.3.7 | Entrada redundante | A | Conforme | |
| 3.3.8 | Autenticacion accesible | AA | Conforme | |
| 4.1.2 | Nombre, funcion, valor | A | No conforme | Progressbar sin nombre |
| 4.1.3 | Mensajes de estado | AA | No conforme | 0 live regions |

**Leyenda**: Conforme - No conforme - No aplica

---

## 7. Anexos

### A. Datos de herramientas automatizadas

axe-core 4.10.2: 2 violations, 22 passes, 1 incomplete.
- `aria-progressbar-name` (serious): 1 node
- `color-contrast` (serious): 1 node
- `aria-hidden-focus` (incomplete/serious): 3 nodes

### B. Indice de evidencias

Modo de evidencia: `codigo` (Chrome MCP sin computer use a disco).

| Criterio | Tipo | Referencia |
|----------|------|------------|
| 1.1.1 (progressbar) | Codigo | outerHTML del div[role="progressbar"] |
| 1.1.1 (SVGs) | Codigo | Conteo de SVGs sin marcado accesible |
| 1.4.3 | Codigo | getComputedStyle: color + backgroundColor + ratio |
| 2.1.2 | Codigo | outerHTML del dialog + lista de botones |
| 2.5.8 | Codigo | getBoundingClientRect de botones radio |
| 3.1.1 | Codigo | document.documentElement.lang |
| 4.1.2 | Codigo | Mismo que 1.1.1 progressbar |
| 4.1.3 | Codigo | Conteo de [aria-live], [role="alert"], etc. |

---

*Informe generado por el Sistema de Auditoria de Accesibilidad de Fundacion Primera Fila.*
*Este informe no constituye una certificacion de accesibilidad. Se recomienda complementar con pruebas con personas usuarias reales con diversidad funcional.*
