# Fuente ÚNICA del código de la Sesión 1. De aquí salen:
#   - el notebook para Colab                    (notebook.py genera 1)
#   - las salidas que muestran las diapositivas (notebook.py ejecuta 1 → salidas_sesion1.json)
#
# No es un script ejecutable: es el formato «percent» de Jupyter.
#   # %% [markdown]            celda de texto (líneas con «# »)
#   # %% <id> [banderas]       celda de código
# Banderas:
#   generador   en el notebook se antepone el código de descarga_datos.py, ayudas.py y
#               ayudas_sesion1.py (celda oculta)
#   solucion    solución de un ejercicio: se ejecuta para verificar, NO va al notebook
#   plantilla   ejercicio por completar: va al notebook, NO se ejecuta al verificar
#   precalculo  cifras que las diapositivas citan en el texto: se ejecuta, NO va al notebook

# %% [markdown]
# # Sesión 1 · Del dato al modelo, del equipo a la nube
# **Machine Learning y Seguridad en la Nube — Especialización en Big Data (UCompensar)**
#
# Este notebook acompaña las [diapositivas de la sesión](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/diapositivas/sesion-1-del-dato-al-modelo.html).
# Cada celda de código lleva entre corchetes el mismo identificador que el código proyectado
# (por ejemplo, `# [cargar]`), así que puedes seguir la clase celda a celda. La portada del curso
# está en https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/
#
# 1. **Archivo → Guardar una copia en Drive**, para que tus cambios no se pierdan.
# 2. Ejecuta las celdas **en orden** con **Mayús + Enter**: varias usan resultados de las anteriores.
#    Si Colab avisa que el notebook no es de Google, elige **Ejecutar de todos modos**.
# 3. Si Colab se reinicia o una celda falla al repetirla, usa **Entorno de ejecución → Ejecutar anteriores**.
# 4. La primera celda descarga los datos de *Yarumo Fibra S.A.S.*, un operador de internet **ficticio**:
#    sus 8000 clientes se inventaron con una semilla fija y no corresponden a personas ni a empresas reales.
#
# En el texto y en los mensajes de revisión, los decimales van con coma (23,6 %); en el código y en
# lo que calcula Python, con punto (`0.2355`).

# %% datos generador
# Descarga datos/clientes.csv (los 8000 clientes de Yarumo) y comprueba su huella SHA-256
ruta = descargar_datos("datos")

# %% [markdown]
# ## 1 · Repaso exprés de Colab y pandas
#
# Un notebook tiene dos clases de celdas: las de **texto**, como esta, y las de **código**. Una celda
# de código se ejecuta con **Mayús + Enter**; lo que escribe `print(...)` aparece debajo, y si la
# última línea es un valor, Colab también lo muestra. Lo que va después de `#` es un comentario:
# Python lo ignora y nosotros lo usamos para explicar.
#
# **pandas** es la biblioteca de tablas de Python. Una **serie** es una columna; un **DataFrame**, una
# tabla con columnas que tienen nombre. La celda siguiente tiene el truco que más usaremos hoy.

# %% repaso
import pandas as pd          # pandas: la biblioteca de tablas; «pd» es su apodo

# Una serie (una columna) con cinco respuestas inventadas
respuestas = pd.Series(["Sí", "No", "No", "Sí", "No"])

# Comparar devuelve True o False en cada fila...
es_si = respuestas == "Sí"
print(es_si.tolist())

# ...y como True vale 1 y False vale 0, el promedio es la proporción de «Sí»
print("Proporción de «Sí»:", es_si.mean())

# %% cargar
# Leer el archivo que dejó la primera celda: una fila por cliente
clientes = pd.read_csv(ruta)
clientes.shape          # (filas, columnas)

# %% vistazo
# pandas oculta columnas cuando hay más de 20: le pedimos que las muestre todas
pd.set_option("display.max_columns", None)

# Las primeras cinco filas. Desliza la tabla a la derecha para ver las 21 columnas
clientes.head()

# %% [markdown]
# ### Diccionario de datos
#
# Son los clientes **activos el 30 de junio de 2026**, con lo que Yarumo sabía de cada uno en esa
# fecha. La tabla está **seudonimizada**: el nombre, la cédula, la dirección y el teléfono se
# quedaron en el sistema de gestión de clientes (CRM) y solo viaja el seudónimo `id_cliente`. Aun
# así, la edad, la ciudad y el estrato siguen siendo **datos personales**, porque Yarumo puede volver
# a unirlos con la persona.
#
# | Columna | Qué registra | Valores o unidad |
# |---|---|---|
# | `id_cliente` | Seudónimo del cliente | C00001… |
# | `ciudad` | Ciudad del servicio | Bogotá, Medellín, Cali, Barranquilla, Pereira |
# | `estrato` | Estrato de la **dirección del servicio** (lo asigna el municipio a la vivienda; no es el ingreso) | 1 a 6 |
# | `edad` | Edad del titular | años |
# | `tipo_cliente` | Hogar o empresa pequeña | Hogar, Pyme |
# | `usuarios` | Personas que usan la conexión (declarado al contratar) | conteo |
# | `tecnologia` | Tecnología de acceso | Fibra, Inalámbrica |
# | `plan_mbps` | Velocidad contratada | 100, 300, 500, 900 Mbps |
# | `contrato` | Permanencia pactada | Mes a mes, 12 meses, 24 meses |
# | `antiguedad_meses` | Meses como cliente | 1 a 96 |
# | `tv`, `movil` | Si tiene también TV o línea móvil con Yarumo | Sí / No |
# | `medio_pago` | Cómo paga | Débito automático, Tarjeta de crédito, Pago en línea, Efectivo |
# | `descuento` | Si tiene un descuento vigente (el de bienvenida dura 12 meses; algunos clientes antiguos tienen otras promociones) | Sí / No |
# | `factura_mensual` | Valor mensual facturado | pesos (COP), ficticios |
# | `consumo_gb` | Datos consumidos al mes, en promedio | GB |
# | `velocidad_pct` | Velocidad medida como % de la contratada (promedio de 3 meses) | % |
# | `caidas_3m` | Caídas del servicio en los últimos 3 meses | conteo |
# | `llamadas_soporte_3m` | Llamadas a soporte en los últimos 3 meses | conteo |
# | `pagos_tardios_12m` | Facturas pagadas tarde en los últimos 12 meses | conteo |
# | `cancelo` | **Etiqueta**: si canceló entre el 1 de julio y el 30 de septiembre de 2026 | Sí / No |

# %% conteo
# ¿Cuántos clientes tiene cada tipo de contrato? value_counts() cuenta las categorías
clientes["contrato"].value_counts()

# %% [markdown]
# ## 2 · De la frecuencia a la probabilidad
#
# La **frecuencia relativa** de un evento (qué parte de los clientes canceló) es nuestra primera
# estimación de su **probabilidad**. Para calcularla usamos el truco del repaso: una columna de
# 1 y 0, y su promedio.

# %% tasa
# Una columna numérica: 1 si el cliente canceló, 0 si no
clientes["cancelo_num"] = (clientes["cancelo"] == "Sí").astype(int)

# Frecuencia relativa: el promedio de una columna de 0 y 1 es la proporción de 1
n_cancelaron = clientes["cancelo_num"].sum()
p_cancela = clientes["cancelo_num"].mean()
print(f"Cancelaron {n_cancelaron} de {len(clientes)} clientes")
print(f"P(cancela) ≈ {p_cancela:.4f}")      # :.4f escribe el número con 4 decimales

# %% [markdown]
# La **probabilidad condicional** $P(\text{cancela} \mid \text{mes a mes})$ se lee «probabilidad de que
# cancele, **dado que** tiene contrato mes a mes». Se calcula igual, pero **solo dentro de ese grupo**.

# %% condicional
# P(cancela | mes a mes): nos quedamos SOLO con los clientes mes a mes...
solo_mes_a_mes = clientes[clientes["contrato"] == "Mes a mes"]

# ...y dentro de ese grupo calculamos la misma proporción
n_grupo = len(solo_mes_a_mes)
n_cancelan = solo_mes_a_mes["cancelo_num"].sum()
p_cond = solo_mes_a_mes["cancelo_num"].mean()
print(f"Mes a mes: {n_cancelan} de {n_grupo} cancelaron")
print(f"P(cancela | mes a mes) ≈ {p_cond:.4f}")

# %% inversa
# Al revés: entre los que cancelaron, ¿qué parte tenía contrato mes a mes?
se_fueron = clientes[clientes["cancelo_num"] == 1]
p_inv = (se_fueron["contrato"] == "Mes a mes").mean()
print(f"P(mes a mes | cancela) ≈ {p_inv:.4f}")

# %% [markdown]
# ---
# # Práctica
#
# Tres ejercicios. Cada uno tiene una **celda de revisión** que te dice qué está bien y, si algo
# falla, te da una pista sin darte la respuesta. Puedes corregir y volver a revisar las veces que
# quieras. Si una celda da un error que menciona `Ellipsis` o `ellipsis`, te quedó un `...` sin
# reemplazar; si dice `NameError`, falta ejecutar sin errores una celda anterior.
#
# ## Ejercicio 1 · La tasa de cancelación de cada tipo de contrato
#
# Calcula, para cada tipo de contrato (`Mes a mes`, `12 meses` y `24 meses`), qué parte de sus
# clientes canceló: $P(\text{cancela} \mid \text{contrato})$. Guárdala en `tasa_contrato`.
#
# **Pistas:**
#
# - `groupby` repite una cuenta en cada grupo. El patrón es
#   `tabla.groupby("columna de los grupos")["columna que se promedia"].mean()`.
# - La columna que se promedia es la de 1 y 0: mira cómo se llama en `[tasa]`. Si ves el error
#   «agg function failed», estás promediando una columna de texto.
# - También vale hacerlo grupo por grupo, como en `[condicional]`, y juntar las tres cifras en un
#   diccionario.

# %% ej1 plantilla
# Ejercicio 1 · Reemplaza los ... por tu cálculo y ejecuta la celda
tasa_contrato = ...
tasa_contrato

# %% ej1_solucion solucion
tasa_contrato = clientes.groupby("contrato")["cancelo_num"].mean()
tasa_contrato

# %% ej1_revision
# Revisa tu tasa_contrato (si ves «todavía no existe», ejecuta antes la celda del ejercicio)
revisar_tasa_contrato()

# %% [markdown]
# ### La misma cuenta, para cada mes de antigüedad
#
# Con el patrón del ejercicio 1, ahora los grupos son los meses que lleva cada cliente con Yarumo.
# La línea punteada es la tasa de todos los clientes. ¿Qué pasa en los primeros meses? ¿Dónde queda
# la tasa después del segundo año?

# %% antiguedad
# Tasa de cancelación de cada mes de antigüedad
por_mes = clientes.groupby("antiguedad_meses")["cancelo_num"].mean()

# .loc[1:36] se queda con los meses 1 a 36; grafica_por_mes es una ayuda del curso que dibuja
grafica_por_mes(por_mes.loc[1:36], p_cancela)

# %% [markdown]
# ## 3 · Características y etiqueta
#
# Para que un modelo aprenda hay que separar dos cosas:
#
# - la **etiqueta** (`y`): lo que queremos predecir; aquí, si el cliente canceló (1) o no (0);
# - las **características** (`X`): lo que el modelo puede mirar para predecir. Solo valen datos que
#   Yarumo **ya conocía el 30 de junio**, antes de saber quién cancelaría.
#
# Hoy usamos a propósito solo dos características; en la sesión 2 usaremos muchas más.

# %% xy
# La etiqueta: lo que queremos predecir (1 = canceló, 0 = no canceló)
y = clientes["cancelo_num"]

# Las características: dos columnas que Yarumo conocía el 30 de junio.
# (== "Mes a mes") da True o False; .astype(int) lo vuelve 1 o 0
X = pd.DataFrame({
    "mes_a_mes": (clientes["contrato"] == "Mes a mes").astype(int),
    "antiguedad_meses": clientes["antiguedad_meses"],
})
print("X:", X.shape, "  y:", y.shape)
X.head(3)

# %% [markdown]
# ## Ejercicio 2 · Separa a los clientes en entrenamiento y prueba
#
# El modelo aprende con unos clientes (**entrenamiento**) y se examina con otros que **no vio**
# (**prueba**). Completa la llamada a `train_test_split` para que:
#
# - la **cuarta parte** de los clientes quede para prueba;
# - los dos grupos tengan **la misma tasa de cancelación** que la tabla completa (eso es
#   *estratificar*: dile a la función qué columna debe conservar su proporción);
# - la división sea **reproducible**: usa la semilla del curso, `2026`, para que todos tengamos
#   los mismos clientes en cada grupo.

# %% ej2 plantilla
from sklearn.model_selection import train_test_split

# Ejercicio 2 · Reemplaza los tres ... (uno por línea)
X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X, y,
    test_size=...,      # ← la fracción de clientes que va a prueba
    stratify=...,       # ← la columna cuya proporción hay que conservar
    random_state=...,   # ← la semilla del curso
)
print("Entrenamiento:", X_ent.shape, "  Prueba:", X_prueba.shape)

# %% ej2_solucion solucion
from sklearn.model_selection import train_test_split

X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=2026,
)
print("Entrenamiento:", X_ent.shape, "  Prueba:", X_prueba.shape)

# %% ej2_revision
# Revisa X_ent, X_prueba, y_ent y y_prueba
revisar_division()

# %% [markdown]
# ## Ejercicio 3 · La línea base contra un primer modelo
#
# Antes de celebrar un modelo hay que compararlo con una **línea base**: una regla que no piensa.
# La nuestra (`DummyClassifier`) dice siempre lo más frecuente, «no cancela». El primer modelo es un
# **árbol de decisión** con solo dos niveles de preguntas (`max_depth=2`). Cómo funciona un árbol
# por dentro lo vemos en la sesión 2; hoy importa cómo se compara.
#
# En scikit-learn todo modelo se usa igual: se **crea**, se **entrena** con `fit` y **predice** con
# `predict`. La celda ya trae todo lo de la línea base. Completa lo del árbol **imitando cada línea
# de la base**. Antes de ejecutar, apunta tu predicción: ¿acertará más el árbol? ¿Encontrará a más de
# los que cancelaron?
#
# El paso 6 es opcional y no se resuelve imitando: piénsalo.

# %% ej3 plantilla
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

# 1 · Crear los dos modelos (random_state: el árbol sale igual cada vez)
base = DummyClassifier(strategy="most_frequent")              # siempre «no cancela»
arbol = DecisionTreeClassifier(max_depth=2, random_state=2026)

# 2 · Entrenar: cada modelo aprende con los clientes de entrenamiento
base.fit(X_ent, y_ent)
arbol.fit(..., ...)                    # ← ¿con qué datos aprende?

# 3 · Predecir para los clientes de prueba, que ninguno de los dos vio
pred_base = base.predict(X_prueba)
pred_arbol = ...                       # ← imita la línea de arriba

# 4 · Exactitud: la proporción de aciertos en prueba
exactitud_base = (pred_base == y_prueba).mean()
exactitud_arbol = ...                  # ← imita la línea de arriba

# 5 · Encontrados: predijo «cancela» (1) Y el cliente de verdad canceló (1).
#     & significa «y, a la vez»; .sum() cuenta los True
encontrados_base = ((pred_base == 1) & (y_prueba == 1)).sum()
encontrados_arbol = ...                # ← imita la línea de arriba

# 6 · (Opcional) Falsas alarmas: el árbol predijo «cancela» (1), pero el cliente NO canceló
falsas_alarmas_arbol = ...

# %% ej3_solucion solucion
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

base = DummyClassifier(strategy="most_frequent")
arbol = DecisionTreeClassifier(max_depth=2, random_state=2026)

base.fit(X_ent, y_ent)
arbol.fit(X_ent, y_ent)

pred_base = base.predict(X_prueba)
pred_arbol = arbol.predict(X_prueba)

exactitud_base = (pred_base == y_prueba).mean()
exactitud_arbol = (pred_arbol == y_prueba).mean()

encontrados_base = ((pred_base == 1) & (y_prueba == 1)).sum()
encontrados_arbol = ((pred_arbol == 1) & (y_prueba == 1)).sum()

falsas_alarmas_arbol = ((pred_arbol == 1) & (y_prueba == 0)).sum()

# %% ej3_revision
# Revisa el árbol, las dos exactitudes, los encontrados y (si la hiciste) las falsas alarmas
revisar_modelos()

# %% [markdown]
# ### Las dos comparaciones, lado a lado

# %% trampa
# Izquierda: exactitud en prueba. Derecha: de los que cancelaron, a cuántos encontró cada uno
grafica_trampa()

# %% [markdown]
# ### La regla que aprendió el árbol
#
# Un árbol de dos niveles es una regla que se puede leer: cada `|---` es una pregunta sobre una
# característica, y cada `class` es lo que predice al final de esa rama (0 = no cancela,
# 1 = cancela). Como `mes_a_mes` solo vale 0 o 1, `mes_a_mes <= 0.50` quiere decir «no es mes a
# mes»; y `antiguedad_meses <= 14.50`, «14 meses o menos». Nadie le escribió esta regla: la sacó
# de los 6000 clientes de entrenamiento. Compárala con la gráfica de la antigüedad.

# %% regla
from sklearn.tree import export_text

# La regla del árbol, en texto (0 = no cancela, 1 = cancela).
# Si ves NotFittedError, completa antes el ejercicio 3
print(export_text(arbol, feature_names=list(X.columns)))

# %% [markdown]
# ### Para la discusión
#
# 1. La gerencia solo mira la exactitud. Con tus cifras, ¿qué le dirías?
# 2. El árbol marcó como «cancela» a algunos clientes que **no** cancelaron. Si a cada uno de los
#    marcados se le ofrece un descuento, ¿qué le cuesta a Yarumo cada falsa alarma? ¿Y cada cliente
#    que el modelo no encontró?
# 3. En el ejercicio 1, la tasa de cancelación mes a mes salió mucho mayor que la de 24 meses.
#    ¿Prueba eso que el contrato causa la cancelación? ¿Qué otra diferencia entre esos clientes podría explicarlo? (Pista: mira
#    la antigüedad.)
# 4. Este notebook corre en una máquina de Google, en la nube. ¿Subirías aquí la tabla con el nombre
#    y la cédula de los clientes? ¿Por qué?
#
# *Doble clic para escribir aquí tus respuestas.*

# %% responsabilidad precalculo
# Diagrama de la diapositiva de responsabilidad compartida (esta celda no va al notebook).
# Matriz de Microsoft Learn, «Responsabilidad compartida en la nube» (ms.date 2026-08-24),
# sección #responsibility-matrix, según FUENTES.md (F-AZ-01). C = cliente, X = compartida,
# M = Microsoft. Columnas: Local, IaaS, PaaS, SaaS.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matriz = [
    ("Datos del cliente", "CCCC"),
    ("Opciones y configuración", "CCCC"),
    ("Identidades y usuarios", "CCCC"),
    ("Dispositivos cliente", "CCCX"),
    ("Aplicaciones", "CCXX"),
    ("Controles de red", "CCXM"),
    ("Sistema operativo", "CCMM"),
    ("Hosts físicos", "CMMM"),
    ("Red física", "CMMM"),
    ("Centro de datos físico", "CMMM"),
]
estilo = {"C": ("#8A2E9E", "white", "Cliente"), "X": ("#FAA012", "#2B2B2B", "Compartida"),
          "M": ("#DDE2E7", "#3D4650", "Microsoft")}
columnas = ["Local", "IaaS", "PaaS", "SaaS"]
_estilo()
fig, ax = plt.subplots(figsize=(10.4, 5.6))
ancho, alto, x0 = 1.0, 0.42, 2.7
for j, nombre in enumerate(columnas):
    ax.text(x0 + j * ancho + ancho / 2, 0.3, nombre, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#2B2B2B")
for i, (capa, codigo) in enumerate(matriz):
    yy0 = -(i + 1) * alto - (0.12 if i >= 3 else 0)
    ax.text(x0 - 0.12, yy0 + alto / 2, capa, ha="right", va="center", fontsize=12.5,
            color="#2B2B2B", fontweight="bold" if i < 3 else "normal")
    for j, letra in enumerate(codigo):
        fondo, tinta, texto = estilo[letra]
        ax.add_patch(Rectangle((x0 + j * ancho + 0.03, yy0 + 0.03), ancho - 0.06, alto - 0.06,
                               facecolor=fondo, edgecolor="none"))
        ax.text(x0 + j * ancho + ancho / 2, yy0 + alto / 2, texto, ha="center", va="center",
                fontsize=10.5, color=tinta)
ax.annotate("", xy=(x0 + 4 * ancho + 0.12, -0.04), xytext=(x0 + 4 * ancho + 0.12, -3 * alto + 0.04),
            arrowprops=dict(arrowstyle="-", color="#8A2E9E", linewidth=2))
ax.text(x0 + 4 * ancho + 0.22, -1.5 * alto, "siempre\ndel cliente", ha="left", va="center",
        fontsize=11.5, color="#6A1F7A", fontweight="bold")
ax.set_xlim(0, x0 + 4 * ancho + 1.2)
ax.set_ylim(-(len(matriz)) * alto - 0.2, 0.55)
ax.axis("off")
fig.tight_layout()
plt.show()

# %% cifras precalculo
# Cifras que las diapositivas citan en el texto (esta celda no va al notebook)
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

d = pd.read_csv("datos/clientes.csv")
c = d["cancelo"] == "Sí"
print(f"clientes {len(d)}, columnas {d.shape[1]}, cancelaron {c.sum()} ({_pct(c.mean())}), "
      f"no cancelaron {(~c).sum()} ({_pct((~c).mean())}); linea base en todos {_pct((~c).mean())}")
for contrato in ["Mes a mes", "12 meses", "24 meses"]:
    g = d["contrato"] == contrato
    print(f"contrato {contrato}: {g.sum()} clientes, {(g & c).sum()} cancelaron, "
          f"P(cancela|contrato) {_pct(c[g].mean())} = {c[g].mean():.4f}")
otros = d["contrato"] != "Mes a mes"
print(f"otros contratos: {otros.sum()} clientes, {(otros & c).sum()} cancelaron, "
      f"P {_pct(c[otros].mean())} = {c[otros].mean():.4f}")
mm = d["contrato"] == "Mes a mes"
print(pd.crosstab(np.where(mm, "Mes a mes", "Con permanencia"), d["cancelo"], margins=True).to_string())
print(f"razón P(cancela|mes a mes) / P(cancela) {_decimal(c[mm].mean() / c.mean(), 2)}")
print(f"P(mes a mes | cancela) {_pct(mm[c].mean())} = {mm[c].mean():.4f}; "
      f"P(mes a mes) {_pct(mm.mean())}; veces mes a mes / 24 meses "
      f"{_decimal(c[mm].mean() / c[d['contrato'] == '24 meses'].mean())}")
ina = d["tecnologia"] == "Inalámbrica"
print(f"inalámbrica: {ina.sum()} clientes, P(cancela|inalámbrica) {_pct(c[ina].mean())} = "
      f"{c[ina].mean():.4f}; fibra {_pct(c[~ina].mean())}")
for desde, hasta in [(1, 3), (4, 9), (10, 12), (13, 15), (37, 96)]:
    g = d["antiguedad_meses"].between(desde, hasta)
    print(f"antigüedad {desde}-{hasta}: {g.sum()} clientes, tasa {_pct(c[g].mean())}")
desc = d["descuento"] == "Sí"
g = d["antiguedad_meses"].between(10, 12)
print(f"meses 10-12: con descuento {(g & desc).sum()} clientes, tasa {_pct(c[g & desc].mean())}; "
      f"sin descuento {(g & ~desc).sum()}, tasa {_pct(c[g & ~desc].mean())}; "
      f"con descuento y más de 12 meses: {(desc & (d['antiguedad_meses'] > 12)).sum()}")
ant = d.groupby("contrato")["antiguedad_meses"].mean()
print(f"antigüedad media: mes a mes {_decimal(ant['Mes a mes'])}, 12 meses {_decimal(ant['12 meses'])}, "
      f"24 meses {_decimal(ant['24 meses'])}")
print(f"descuento vigente: {desc.sum()} clientes; antigüedad máxima con descuento "
      f"{d.loc[desc, 'antiguedad_meses'].max()}; tasa con descuento {_pct(c[desc].mean())}, "
      f"sin {_pct(c[~desc].mean())}")
yy = c.astype(int)
XX = pd.DataFrame({"mes_a_mes": mm.astype(int), "antiguedad_meses": d["antiguedad_meses"]})
xe, xp, ye, yp = train_test_split(XX, yy, test_size=0.25, random_state=2026, stratify=yy)
b = DummyClassifier(strategy="most_frequent").fit(xe, ye)
t = DecisionTreeClassifier(max_depth=2, random_state=2026).fit(xe, ye)
pb, pt = b.predict(xp), t.predict(xp)
enc = int(((pt == 1) & (yp == 1)).sum())
print(f"division: entrenamiento {len(xe)}, prueba {len(xp)}, cancelaron en prueba {yp.sum()}, "
      f"tasa ent {_pct(ye.mean())}, tasa prueba {_pct(yp.mean())}")
print(f"base: exactitud prueba {_pct((pb == yp).mean())} ({(pb == yp).sum()} aciertos), encontrados "
      f"{int(((pb == 1) & (yp == 1)).sum())}")
print(f"arbol: exactitud prueba {_pct((pt == yp).mean())} ({(pt == yp).sum()} aciertos), "
      f"exactitud entrenamiento {_pct((t.predict(xe) == ye).mean())}, encontrados {enc} de {yp.sum()} "
      f"({_pct(enc / yp.sum())}), marcados {int(pt.sum())}, falsas alarmas {int(pt.sum()) - enc}, "
      f"no encontrados {yp.sum() - enc}")
print(f"diferencia de exactitud {_decimal(100 * ((pt == yp).mean() - (pb == yp).mean()))} puntos")
sin_nada = ((~mm) | (d["antiguedad_meses"] > 14))
print(f"regla (mes a mes y antigüedad <= 14) en todos: {(~sin_nada).sum()} clientes, "
      f"tasa {_pct(c[~sin_nada].mean())}; resto {_pct(c[sin_nada].mean())}")
print(d[["id_cliente", "contrato", "antiguedad_meses", "velocidad_pct", "factura_mensual",
         "cancelo"]].head(3).to_string(index=False))
