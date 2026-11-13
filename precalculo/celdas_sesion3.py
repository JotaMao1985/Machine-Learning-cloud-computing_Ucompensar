# Fuente ÚNICA del código de la Sesión 3. De aquí salen:
#   - el notebook para Colab                    (notebook.py genera 3)
#   - las salidas y las gráficas que muestran las diapositivas
#                                               (notebook.py ejecuta 3 → salidas_sesion3.json y
#                                                fuente/diapositivas/recursos/s3/<id>.svg)
#
# No es un script ejecutable: es el formato «percent» de Jupyter.
#   # %% [markdown]            celda de texto (líneas con «# »)
#   # %% <id> [banderas]       celda de código
# Banderas:
#   generador   en el notebook se antepone el código de descarga_datos.py, ayudas.py y
#               ayudas_sesion3.py (celda oculta); los datos se descargan (M3)
#   solucion    solución de un ejercicio: se ejecuta para verificar, NO va al notebook
#   plantilla   ejercicio por completar: va al notebook, NO se ejecuta al verificar
#   precalculo  cifras que las diapositivas citan en el texto: se ejecuta, NO va al notebook
#
# Reglas de la sesión (DISENO_S3_S4.md): KMeans siempre con n_init=10 y random_state=2026 (con
# n_init="auto", scikit-learn 1.6 hace una sola corrida y cambian los tamaños); DBSCAN con
# min_samples=20; los grupos se nombran por su perfil, nunca por el número de etiqueta.

# %% [markdown]
# # Sesión 3 · Grupos sin etiqueta
# **Machine Learning y Seguridad en la Nube — Especialización en Big Data (UCompensar)**
#
# Este notebook acompaña las [diapositivas de la sesión](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/diapositivas/sesion-3-grupos-sin-etiqueta.html).
# Cada celda lleva entre corchetes el mismo identificador que el código proyectado, así que puedes seguir
# la clase celda a celda. La portada del curso está en https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/
# y el enunciado del proyecto final, en [su página](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/proyecto.html).
#
# 1. **Archivo → Guardar una copia en Drive** para que tus cambios no se pierdan.
# 2. Ejecuta las celdas **en orden** con **Mayús + Enter**. Varias usan resultados de las anteriores.
#    Si Colab avisa que el notebook no es de Google, elige **Ejecutar de todos modos**.
# 3. Si Colab se reinicia o una celda falla al repetirla, usa **Entorno de ejecución → Ejecutar anteriores**.
#    Las celdas de ejercicio dan error mientras tengan `...` sin reemplazar (si el error menciona `Ellipsis` o
#    `'ellipsis'`, o el verificador dice que quedan puntos suspensivos, te quedó un `...`). Colab se detiene en el primer error: completa los ejercicios en orden, o ejecuta
#    con **Mayús + Enter** solo las celdas que necesites.
# 4. La primera celda **descarga** los datos del caso *Yarumo Fibra S.A.S.*, una empresa **ficticia** (sus
#    datos se generaron para el curso y no corresponden a personas ni a empresas reales), y comprueba que
#    el archivo sea idéntico al del curso. Si la descarga falla, la celda te dice cómo subirlo a mano.
#
# En el código, los decimales van con punto (así los escribe Python); en el texto, con coma. Todo el
# notebook tarda menos de un minuto en ejecutarse.

# %% datos generador
# Descarga datos/clientes.csv: los mismos 8000 clientes de las sesiones 1 y 2
ruta = descargar_datos("datos")

# %% [markdown]
# # Parte 1 · Agrupamiento: ¿qué tipos de clientes tenemos?
#
# ## 1 · Hoy no hay etiqueta
#
# En la sesión 2, cada cliente traía la respuesta (cuánto consumió, si canceló) y el modelo aprendía a
# predecirla: aprendizaje **supervisado**. Hoy la gerente de experiencia del cliente pregunta otra cosa:
# *¿qué tipos de clientes tenemos?* Nadie ha etiquetado a los clientes por tipo, así que no hay una
# respuesta correcta con la cual comparar. Es aprendizaje **no supervisado**: el **agrupamiento**
# (*clustering*) busca grupos de clientes parecidos entre sí y distintos de los demás.
#
# **Repaso: lo nuevo de pandas y numpy que aparece hoy.** No hace falta memorizarlo; vuelve aquí cuando lo veas.
#
# | Escribes | Qué hace |
# |---|---|
# | `modelo.fit_predict(Z)` | Entrena y devuelve, de una vez, el número de grupo de cada fila |
# | `serie.value_counts().sort_index()` | Cuántas filas tiene cada valor, ordenado por el valor |
# | `tabla.groupby(grupos).agg(nuevo=("col", "mean"))` | Una fila por grupo, con una columna `nuevo` que resume `col` |
# | `lambda t: (t == "Pyme").mean()` | Una función de una línea: la proporción de `"Pyme"` en `t` |
# | `tabla.apply(f, axis=1)` | Aplica la función `f` a cada fila |
# | `pd.crosstab(a, b)` | Cuenta cuántas filas hay en cada combinación de `a` y `b` |
# | `arreglo[:, [0, 2]]` | De un arreglo de numpy, todas las filas y las columnas 0 y 2 |
# | `(etiquetas == -1).sum()` | Cuántas etiquetas valen −1: True cuenta como 1 |
# | `etiquetas.max() + 1` | Con etiquetas 0, 1, 2…, cuántos grupos hay |
# | `clientes.loc[mascara, "col"]` | La columna `col`, solo en las filas donde `mascara` es True |
# | `serie.idxmax()` | La etiqueta (el índice) del valor más alto |
# | `max(d, key=d.get)` | La clave del diccionario `d` con el valor más alto |

# %% cargar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

clientes = pd.read_csv("datos/clientes.csv")
# Colores de los grupos; el último, gris, es para el ruido (-1) de DBSCAN
COLORES = np.array(["#8A2E9E", "#F46201", "#566B7F", "#0E7490", "#94A3B8"])
np.set_printoptions(legacy="1.25")   # números sin np.float64(...) al mostrarlos
print(f"scikit-learn {sklearn.__version__}")
clientes.shape

# %% [markdown]
# ## 2 · Parecido es distancia, y la distancia depende de la escala
#
# Agrupamos con tres columnas que dicen **quién es el cliente y cuánto usa la conexión**:
#
# | Columna | Qué registra | Unidad |
# |---|---|---|
# | `edad` | Edad del titular | años |
# | `usuarios` | Personas que usan la conexión (declarado al contratar) | conteo |
# | <code>consumo&#95;gb</code> | Datos consumidos al mes, en promedio | GB |
#
# `cancelo` **no** entra: no buscamos a los que cancelan, sino tipos de clientes. (El diccionario completo
# está en el notebook de la sesión 1.)

# %% columnas
columnas = ["edad", "usuarios", "consumo_gb"]
X = clientes[columnas]
X.agg(["mean", "std", "min", "max"]).round(1)

# %% [markdown]
# Un cliente es «parecido» a otro si está **cerca**: la distancia en línea recta se calcula con las
# diferencias de cada columna, elevadas al cuadrado. Pero el consumo va de 14 a 2500 GB y los usuarios, de 1 a 14: sin escalar, la
# distancia la decide casi solo el consumo. Como con KNN en la sesión 2, **escalamos**: cada columna queda
# en desviaciones estándar (`StandardScaler`).
#
# Una diferencia con la sesión 2: aquí no hay entrenamiento ni prueba, porque no hay una respuesta que
# predecir y medir. El escalador aprende la media y la desviación de los 8000 clientes.

# %% escala
from sklearn.preprocessing import StandardScaler
escalador = StandardScaler()
Z = escalador.fit_transform(X)            # cada columna, en desviaciones
print("Medias:", Z.mean(axis=0).round(2) + 0.0)   # + 0.0: sin «−0»
print("Desviaciones:", Z.std(axis=0).round(2))

# %% [markdown]
# ## 3 · K-Means (k-medias)
#
# Imagina que Yarumo abre **k puntos de atención** en una ciudad (un mapa de perfiles: sus ejes son la edad,
# los usuarios y el consumo, no calles) y cada cliente va al más cercano. Después,
# cada punto se muda al **centro** de los clientes que le tocaron, y se repite: los clientes cambian de punto
# si otro les quedó más cerca, los puntos se vuelven a mudar… hasta que nadie cambia. Cada punto es un
# **centro del grupo** (centroide) y sus clientes, un **grupo** (*cluster*).
#
# La celda siguiente lo hace a mano con 300 clientes y solo dos columnas (edad y consumo, escaladas), para
# poder dibujarlo. Si la lees, verás las dos operaciones: `asignar` (cada cliente al centro más cercano) y
# `mudar` (cada centro al promedio de sus clientes).

# %% pasos
def asignar(puntos, centros):
    """El número del centro más cercano a cada punto."""
    distancias = ((puntos[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
    return distancias.argmin(axis=1)

def mudar(puntos, cercano, k):
    """Cada centro pasa al promedio de los puntos que le tocaron."""
    return np.array([puntos[cercano == j].mean(axis=0) for j in range(k)])

def dibujar(ax, puntos, cercano, centros, titulo):
    color = "#94A3B8" if cercano is None else COLORES[cercano]
    ax.scatter(puntos[:, 0], puntos[:, 1], s=9, c=color)
    ax.scatter(centros[:, 0], centros[:, 1], s=180, marker="X",
               c="#1E293B", edgecolors="white")
    ax.set_title(titulo, fontsize=13)

muestra = Z[:300][:, [0, 2]]               # edad y consumo, escalados
rng = np.random.default_rng(2026)
centros = muestra[rng.choice(300, size=3, replace=False)]  # al azar
fig, ejes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True,
                         sharey=True)
dibujar(ejes[0], muestra, None, centros, "1 · Tres centros al azar")
cercano = asignar(muestra, centros)
dibujar(ejes[1], muestra, cercano, centros, "2 · Cada uno al más cercano")
centros = mudar(muestra, cercano, 3)
dibujar(ejes[2], muestra, cercano, centros, "3 · Los centros se mudan")
for vuelta in range(10):                   # repetir (aquí bastan diez vueltas)
    cercano = asignar(muestra, centros)
    centros = mudar(muestra, cercano, 3)
dibujar(ejes[3], muestra, cercano, centros, "4 · Diez vueltas después")
ejes[0].set_ylabel("Consumo (escalado)")
for ax in ejes:
    ax.set_xlabel("Edad (escalada)")
plt.show()

# %% [markdown]
# K-Means busca los centros que dejan la menor **inercia**: la suma de los **cuadrados** de las distancias
# de cada cliente a su centro. El resultado depende de dónde empiecen los centros, así que scikit-learn
# arranca con *k-means++* (centros iniciales lejos unos de otros), **repite** el proceso `n_init` veces y
# se queda con la corrida de menor inercia. Ojo: en la versión de Colab, `n_init="auto"` hace **una sola**
# corrida; por eso hoy siempre escribimos `n_init=10` y la semilla `random_state=2026`.

# %% kmeans4
from sklearn.cluster import KMeans
km4 = KMeans(n_clusters=4, n_init=10, random_state=2026)
grupos = km4.fit_predict(Z)           # el número de grupo de cada cliente
print(f"Inercia: {km4.inertia_:.1f}")
pd.Series(grupos).value_counts().sort_index()

# %% [markdown]
# Los números de los grupos (0, 1, 2, 3) **no significan nada**: son solo nombres. Con otra semilla o con
# otra versión de scikit-learn, el mismo grupo puede salir con otro número. Por eso, más adelante, cada
# grupo se nombra por su **perfil**.
#
# ## 4 · ¿Cuántos grupos?
#
# K-Means no elige k: hay que dárselo. Dos ayudas:
#
# - **El método del codo** (*elbow method*). La inercia siempre baja al subir k (con 8000 grupos sería 0). Se busca el punto donde deja
#   de bajar mucho. Es una **heurística**, no una regla.
# - **El coeficiente de silueta.** Para cada cliente compara *a*, su distancia media a los de su grupo, con *b*, su distancia
#   media al grupo vecino más cercano: $s = (b - a) / \max(a, b)$. Va de −1 a 1; cerca de 1, el cliente está
#   bien metido en su grupo; cerca de 0, está en la frontera. La silueta del agrupamiento es el promedio.
#
# La **línea base** es repartir a los clientes **al azar** en 4 grupos: su silueta queda cerca de 0.

# %% codo_silueta
from sklearn.metrics import silhouette_score
ks = range(2, 9)
inercias, siluetas = [], []
for k in ks:
    modelo = KMeans(n_clusters=k, n_init=10, random_state=2026).fit(Z)
    inercias.append(modelo.inertia_)
    siluetas.append(silhouette_score(Z, modelo.labels_))
azar = np.random.default_rng(2026).integers(0, 4, size=len(Z))
silueta_azar = silhouette_score(Z, azar)  # línea base: 4 grupos al azar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
ax1.plot(ks, inercias, "o-", color="#8A2E9E", linewidth=2.5)
ax1.set_title("El codo: la inercia baja cada vez menos")
ax1.set_xlabel("Número de grupos (k)")
ax1.set_ylabel("Inercia")
ax2.plot(ks, siluetas, "o-", color="#F46201", linewidth=2.5,
         label="K-Means")
ax2.axhline(silueta_azar, color="#566B7F", linestyle="--",
            label="4 grupos al azar")
ax2.set_title("La silueta: más alta, grupos más separados")
ax2.set_xlabel("Número de grupos (k)")
ax2.set_ylabel("Silueta media")
ax2.legend(frameon=False)
plt.show()

# %% [markdown]
# ### La trampa de no escalar
#
# ¿Y si no escalamos? Antes de ejecutar la celda, predice: **¿la silueta sube o baja?**

# %% sin_escalar
km_crudo = KMeans(n_clusters=4, n_init=10, random_state=2026)
grupos_crudos = km_crudo.fit_predict(X)        # sin escalar
print(f"Silueta sin escalar: {silhouette_score(X, grupos_crudos):.3f}")
print(f"Silueta escalada:    {silhouette_score(Z, grupos):.3f}")
X.groupby(grupos_crudos).agg(["min", "max"]).round(0)

# %% [markdown]
# Sin escalar, la silueta **sube**, y los grupos son peores: casi solo ordenan por consumo (cada grupo es
# una franja de GB), y sus clientes no se parecen entre sí en edad ni en usuarios. La silueta mide
# distancias con las columnas que le das; no sabe si esas columnas son comparables. **La silueta ayuda,
# pero no decide.**
#
# ## 5 · Los cuatro grupos, por su perfil
#
# Para la gerente, un grupo sirve si se puede **describir**. Resumimos cada uno con sus promedios.

# %% perfiles
perfiles = clientes.groupby(grupos).agg(
    clientes=("edad", "size"),
    edad=("edad", "mean"),
    usuarios=("usuarios", "mean"),
    consumo_gb=("consumo_gb", "mean"),
    pymes=("tipo_cliente", lambda t: (t == "Pyme").mean()))
perfiles.round({"edad": 1, "usuarios": 1, "consumo_gb": 0, "pymes": 3})

# %% [markdown]
# Cada grupo se nombra por su perfil, no por su número. La función `nombrar` lo hace con reglas que
# cualquiera puede leer: si cambia el número, el nombre no cambia. Describimos clientes, no juzgamos
# personas.

# %% nombres
def nombrar(fila):
    """Un nombre por el perfil del grupo, no por su número."""
    if fila["pymes"] > 0.5:
        return "Pymes con muchos usuarios"
    if fila["edad"] >= 55:
        return "Titulares mayores, 1 o 2 personas"
    if fila["usuarios"] >= 3:
        return "Hogares de 3 a 6 personas"
    return "Titulares jóvenes, hogares pequeños"

nombre = perfiles.apply(nombrar, axis=1)
nombre

# %% [markdown]
# **¿Cancelan igual?** `cancelo` no entró al agrupamiento. Si los grupos cancelan distinto, el agrupamiento
# encontró algo que la gerente puede usar. Lo calculas tú en el ejercicio 2.
#
# ## 6 · Agrupamiento jerárquico
#
# El **jerárquico aglomerativo** empieza con cada cliente solo y une, paso a paso, los dos grupos más
# cercanos, hasta que queda uno. El **dendrograma** dibuja esas uniones: la altura de cada unión es qué tan
# distintos eran los grupos que se unieron. **Cortarlo a una altura** da k grupos. Con el criterio de
# **Ward**, cada unión es la que menos aumenta la suma de cuadrados dentro de los grupos, un objetivo
# parecido al de K-Means. Lo dibujamos con 200 clientes al azar: con 8000 no se leería.

# %% jerarquico
from scipy.cluster.hierarchy import (dendrogram, linkage,
                                     set_link_color_palette)
rng = np.random.default_rng(2026)
muestra200 = Z[rng.choice(len(Z), size=200, replace=False)]
uniones = linkage(muestra200, method="ward")   # de abajo hacia arriba
corte = (uniones[-4, 2] + uniones[-3, 2]) / 2  # la altura que deja 4 grupos
set_link_color_palette(list(COLORES[:4]))
fig, ax = plt.subplots(figsize=(11, 3.9))
dendrogram(uniones, no_labels=True, color_threshold=corte,
           above_threshold_color="#94A3B8", ax=ax)
ax.axhline(corte, color="#1E293B", linestyle="--", linewidth=1)
ax.set_title("Cortar el dendrograma a una altura da 4 grupos")
ax.set_ylabel("Altura de la unión (Ward)")
ax.set_xlabel("200 clientes al azar")
plt.show()

# %% [markdown]
# El criterio de **enlace** (*linkage*) para unir importa. El **enlace simple** (*single*) une los dos grupos
# que tienen los clientes más cercanos entre sí. Antes de ejecutar, predice: con los 8000 clientes y 4 grupos,
# **¿cuántos clientes tendrá el grupo más grande con el enlace simple?**

# %% encadena
from sklearn.cluster import AgglomerativeClustering
for enlace in ["ward", "single"]:
    jerarquico = AgglomerativeClustering(n_clusters=4, linkage=enlace)
    etiquetas_j = jerarquico.fit_predict(Z)
    tamanos = sorted(np.bincount(etiquetas_j).tolist(), reverse=True)
    print(f"{enlace:>6}: {tamanos}")

# %% [markdown]
# Con el enlace simple, basta una cadena de clientes parecidos para pegarlo todo: el grupo más grande se queda
# con casi todos. Ward da tamaños parecidos.
#
# ## 7 · DBSCAN: grupos por densidad, y el ruido
#
# Piensa en una plaza llena. En el centro de una zona densa, cada persona tiene, a menos de un paso (`eps`),
# al menos `min_samples` personas cerca, contándose a sí misma; quien está en el borde entra al grupo si
# toca una zona densa, y quien está solo, lejos de todos, es **ruido**. DBSCAN no pide k: lo
# encuentra. Y no obliga a nadie a entrar en un grupo: el ruido sale con la etiqueta `-1`.
#
# Primero, un ejemplo de juguete: dos **lunas** entrelazadas (datos inventados con `make_moons`).

# %% lunas
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
lunas, _ = make_moons(n_samples=400, noise=0.08, random_state=2026)
por_kmeans = KMeans(n_clusters=2, n_init=10,
                    random_state=2026).fit_predict(lunas)
por_dbscan = DBSCAN(eps=0.25, min_samples=20).fit_predict(lunas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
ax1.scatter(lunas[:, 0], lunas[:, 1], s=12, c=COLORES[por_kmeans])
ax1.set_title("K-Means parte las lunas por la mitad")
ax2.scatter(lunas[:, 0], lunas[:, 1], s=12, c=COLORES[por_dbscan])
ax2.set_title("DBSCAN sigue cada luna")
plt.show()

# %% [markdown]
# K-Means supone grupos redondos alrededor de un centro; DBSCAN sigue la forma que tenga la zona densa.
# (En `COLORES[etiquetas]`, el ruido, `-1`, toma el último color: el gris.)
#
# ¿Y en Yarumo? Con nuestras tres columnas escaladas, `eps=0.5` y `min_samples=20`. Antes de ejecutar,
# predice: **¿quiénes serán el ruido?**

# %% dbscan
db = DBSCAN(eps=0.5, min_samples=20)
etiquetas = db.fit_predict(Z)
ruido = etiquetas == -1                     # True si quedó como ruido
print(f"Grupos: {etiquetas.max() + 1} · ruido: {ruido.sum()} clientes")
clientes.groupby(ruido)[columnas].mean().round(1).rename(
    index={False: "el resto", True: "el ruido"})

# %% [markdown]
# Con estos datos, DBSCAN no sirve para hacer segmentos: deja casi a todos en un solo grupo. Sirve para otra
# cosa: **encontrar lo atípico**, los clientes que no se parecen a nadie. Con `eps` más pequeño pasa algo curioso:
# lo ves en el ejercicio 3.
#
# **El método, las columnas y k se eligen por la decisión.** La silueta, el codo y el dendrograma ayudan;
# ninguno decide solo.
#
# ---
# ## Ejercicio 1 · ¿Cuántos grupos? (en clase)
#
# Calcula la **silueta** de K-Means para k de 2 a 8, con los datos **escalados** (`Z`), `n_init=10` y la
# semilla 2026. Entrega:
#
# - `siluetas`: el diccionario con la silueta de cada k;
# - `k_silueta`: el k con la silueta más alta;
# - `k_gerente`: el k que le recomiendas a la gerente, sabiendo que **puede atender de 3 a 5 ofertas
#   distintas** (una por grupo).
#
# Pista: la celda [codo_silueta]. Cambia cada `...`; el verificador te dice qué está bien y, en lo demás, te
# da una pista. Después justifica tu `k_gerente` en «Tus respuestas»: ¿qué te dicen el codo, la silueta y la
# gerente?

# %% ej1 plantilla
siluetas = {}
for k in range(2, 9):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=2026)
    etiquetas_k = modelo.fit_predict(...)      # con los datos escalados
    siluetas[k] = ...                          # silhouette_score(...)
    print(k, siluetas[k])
k_silueta = ...           # el k de la silueta más alta (máximo del diccionario)
k_gerente = ...                                # el que le recomiendas
revisar_k(siluetas=siluetas, k_silueta=k_silueta, k_gerente=k_gerente)

# %% ej1_solucion solucion
siluetas = {}
for k in range(2, 9):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=2026)
    etiquetas_k = modelo.fit_predict(Z)
    siluetas[k] = silhouette_score(Z, etiquetas_k)
    print(k, round(siluetas[k], 3))
k_silueta = max(siluetas, key=siluetas.get)
k_gerente = 4
revisar_k(siluetas=siluetas, k_silueta=k_silueta, k_gerente=k_gerente)

# %% [markdown]
# ## Ejercicio 2 · ¿Cancelan igual? (en clase)
#
# Con los cuatro grupos de la clase (`grupos`, de la celda [kmeans4]), calcula la **tasa de cancelación de
# cada grupo**: la proporción de clientes con `cancelo == "Sí"`. Entrega el **número del grupo** que más
# cancela y **su tasa** (como proporción, entre 0 y 1). Pista: en la sesión 1, la tasa era el promedio de una
# columna de True y False; aquí, ese promedio se hace por grupo (`.groupby(grupos).mean()`).
#
# Después responde abajo: ¿qué **perfil** tiene ese grupo (míralo en [perfiles] y [nombres])? ¿Qué le
# propondrías a la gerente?

# %% ej2 plantilla
cancela = clientes["cancelo"] == "Sí"         # True si canceló
tasas = ...                                    # la tasa de cada grupo
print(tasas)
grupo_mas = ...                                # el número del que más cancela
tasa_mas = ...                                 # su tasa
revisar_cancelacion(grupos=grupos, grupo=grupo_mas, tasa=tasa_mas)

# %% ej2_solucion solucion
cancela = clientes["cancelo"] == "Sí"
tasas = cancela.groupby(grupos).mean()
print(tasas.round(3))
grupo_mas = tasas.idxmax()
tasa_mas = tasas.max()
revisar_cancelacion(grupos=grupos, grupo=grupo_mas, tasa=tasa_mas)

# %% [markdown]
# ## Ejercicio 3 · Lo atípico (trabajo independiente)
#
# Corre DBSCAN sobre `Z` con `min_samples=20` y dos valores de `eps`: `0.3` y `0.5`. Para cada uno, cuenta
# **cuántos grupos** encontró (el ruido, `-1`, no es un grupo) y **cuántos clientes quedaron como ruido**. Con
# `eps=0.5`, calcula además **qué proporción del ruido son pymes**. Pista: la celda [dbscan].
#
# Después responde abajo: con `eps=0.3`, ¿qué tienen en común los clientes de cada grupo? (Mira
# `pd.crosstab(etiquetas_03, clientes["usuarios"])`.) ¿Por qué pasa eso?

# %% ej3 plantilla
etiquetas_03 = DBSCAN(eps=0.3, min_samples=...).fit_predict(Z)
etiquetas_05 = DBSCAN(eps=..., min_samples=...).fit_predict(Z)
grupos_03 = ...                  # cuántos grupos con eps 0.3, sin el ruido
ruido_03 = ...                   # cuántos clientes son ruido con eps 0.3
grupos_05 = ...
ruido_05 = ...
pymes_ruido = ...                # proporción de pymes en el ruido (eps 0.5)
revisar_ruido(grupos_03=grupos_03, ruido_03=ruido_03, grupos_05=grupos_05,
              ruido_05=ruido_05, pymes_ruido=pymes_ruido)

# %% ej3_solucion solucion
etiquetas_03 = DBSCAN(eps=0.3, min_samples=20).fit_predict(Z)
etiquetas_05 = DBSCAN(eps=0.5, min_samples=20).fit_predict(Z)
grupos_03 = etiquetas_03.max() + 1
ruido_03 = (etiquetas_03 == -1).sum()
grupos_05 = etiquetas_05.max() + 1
ruido_05 = (etiquetas_05 == -1).sum()
pymes_ruido = (clientes.loc[etiquetas_05 == -1, "tipo_cliente"] == "Pyme").mean()
revisar_ruido(grupos_03=grupos_03, ruido_03=ruido_03, grupos_05=grupos_05,
              ruido_05=ruido_05, pymes_ruido=pymes_ruido)

# %% [markdown]
# ## Ejercicio 4 (opcional) · Otra columna
#
# Agrega `plan_mbps` (la velocidad contratada) a las tres columnas, escala de nuevo y agrupa con **k = 5**.
# La tabla cruza los 4 grupos de la clase (filas, por su nombre) con los 5 nuevos (columnas). ¿Cuál de los grupos de antes
# se parte en dos? ¿Por qué se parte? Pista: compara el `plan_mbps` medio de los grupos nuevos. Este
# verificador solo da pistas.

# %% ej4 plantilla
columnas5 = columnas + [...]                   # agrega plan_mbps
Z5 = StandardScaler().fit_transform(clientes[columnas5])
grupos5 = KMeans(n_clusters=..., n_init=10,
                 random_state=2026).fit_predict(Z5)
print(pd.crosstab(nombre[grupos].values, grupos5, rownames=["antes"],
                  colnames=["con plan_mbps"]))
revisar_otra_columna(columnas=columnas5, grupos5=grupos5)

# %% ej4_solucion solucion
columnas5 = columnas + ["plan_mbps"]
Z5 = StandardScaler().fit_transform(clientes[columnas5])
grupos5 = KMeans(n_clusters=5, n_init=10,
                 random_state=2026).fit_predict(Z5)
print(pd.crosstab(nombre[grupos].values, grupos5, rownames=["antes"],
                  colnames=["con plan_mbps"]))
revisar_otra_columna(columnas=columnas5, grupos5=grupos5)

# %% [markdown]
# ---
# # Parte 2 · Nube: riesgos, privacidad y dónde vive el modelo
#
# ## Para Yarumo · Lo que llevas a las salas (sin código)
#
# Esta parte se hace con las diapositivas de la sesión y pasa directo al **plan de nube** del proyecto. Con
# tu pareja, escribe una línea por fila. *Doble clic para escribir aquí.*
#
# | Decisión | Lo que eligen | Por qué, y con qué fuente |
# |---|---|---|
# | Región de Azure para los datos y el modelo | … | … |
# | Base legal: ¿transmisión o transferencia? ¿Con qué contrato? | … | … |
# | Modelo de servicio de nube para el modelo (máquina virtual, App Service o un punto de conexión administrado) | … | … |
# | Riesgo 1 y el control que lo mitiga (de la S2 o de hoy) | … | … |
# | Riesgo 2 y el control que lo mitiga | … | … |
#
# ---
# ### Tus respuestas
#
# *Doble clic para escribir aquí.*
#
# 1. Ejercicio 1: la silueta más alta es la de k = …; le recomiendo a la gerente k = … porque … (¿qué dice el
#    codo?, ¿qué separa el corte de 2 grupos?) …
# 2. Ejercicio 2: el grupo que más cancela es el de … (su perfil, no su número), con una tasa de …; los demás
#    cancelan entre … y … Le propondría a la gerente … ¿Por qué es interesante si `cancelo` no entró al
#    agrupamiento? …
# 3. Ejercicio 3: con `eps` 0,3 salen … grupos y … clientes de ruido; con 0,5, … y … El ruido de 0,5 es sobre
#    todo de … Con 0,3, cada grupo tiene … ¿Por qué? …
# 4. Ejercicio 4 (opcional): con `plan_mbps`, el grupo de … se parte en … y … porque …
#
# ### Autoevaluación (respóndela en el cierre de la sesión)
#
# ¿Verdadero o falso? Explica cada una.
#
# 1. «Con una máquina virtual, Yarumo parchea su sistema operativo». …
# 2. «Si K-Means usa las columnas sin escalar, pasar el consumo de GB a MB no cambia los grupos». …
# 3. «Si Microsoft guarda en Mexico Central los datos de Yarumo, por cuenta de Yarumo y con contrato, pasa a ser
#    el responsable de esos datos». …
# 4. «Una silueta alta no basta para elegir el número de grupos». …

# %% [markdown]
# ---
# ## La respuesta a la gerencia
#
# **¿Qué tipos de clientes tenemos?** Con la edad, los usuarios y el consumo escalados, K-Means encuentra cuatro
# perfiles que se pueden describir: titulares jóvenes de hogares pequeños, hogares de tres a seis personas,
# titulares mayores de hogares de una o dos personas y pymes con muchos usuarios. No cancelan igual, aunque la cancelación no
# entró al agrupamiento. La silueta prefería dos grupos, pero ese corte casi solo aparta a las pymes: el número de
# grupos lo decide lo que la gerente puede hacer con ellos.
#
# **¿Llevamos los datos y el modelo a Azure, y cómo?** Si Microsoft trata los datos por cuenta de Yarumo, es una
# transmisión con contrato; la región se elige mirando dónde quedan los datos; el modelo puede vivir en una
# máquina virtual (IaaS) o en un servicio administrado (PaaS), y cada opción reparte distinto lo que Yarumo
# administra. Según la encuesta de la Cloud Security Alliance de 2026, el riesgo que más pesa hoy es la
# identidad: quién entra y con qué permisos.

# %% cifras precalculo
# Cifras que las diapositivas y sus notas citan (no va al notebook)
import json
import time
from sklearn.metrics import adjusted_rand_score
np.set_printoptions(legacy="1.25")   # que los números salgan sin np.float64(...)
_t0 = time.time()
_cif = {"n": len(clientes)}
_cif["describe"] = X.agg(["mean", "std", "min", "max", "median"]).round(2).to_dict()
_cif["tasa_total"] = round(float(cancela.mean()) * 100, 2)
_cif["pymes_total"] = int((clientes["tipo_cliente"] == "Pyme").sum())
_cif["k"] = {k: {"inercia": round(float(i), 1), "silueta": round(float(siluetas[k]), 4)}
             for k, i in zip(ks, inercias)}   # siluetas: el diccionario del ejercicio 1
_cif["silueta_azar"] = round(float(silueta_azar), 4)
# Sin escalar, para cada k
_cif["silueta_sin_escalar"] = {k: round(float(silhouette_score(
    X, KMeans(n_clusters=k, n_init=10, random_state=2026).fit_predict(X))), 4) for k in ks}
_cif["silueta_k4"] = {"escalada": round(float(silhouette_score(Z, grupos)), 4),
                      "sin_escalar": round(float(silhouette_score(X, grupos_crudos)), 4)}
_cif["sin_escalar_rangos"] = {int(g): {c: [float(X[c][grupos_crudos == g].min()), float(X[c][grupos_crudos == g].max()),
                                           round(float(X[c][grupos_crudos == g].mean()), 1)] for c in columnas}
                              for g in np.unique(grupos_crudos)}
# k = 2: ¿qué aparta?
_g2 = KMeans(n_clusters=2, n_init=10, random_state=2026).fit_predict(Z)
_pyme = (clientes["tipo_cliente"] == "Pyme").to_numpy()
_chico = int(np.argmin(np.bincount(_g2)))
_cif["k2"] = {"tamanos": sorted(map(int, np.bincount(_g2))),
              "pymes_en_el_chico": round(float(_pyme[_g2 == _chico].mean()) * 100, 1),
              "pymes_atrapadas": round(float((_g2[_pyme] == _chico).mean()) * 100, 1),
              "usuarios_chico": round(float(X["usuarios"][_g2 == _chico].mean()), 1),
              "consumo_chico": round(float(X["consumo_gb"][_g2 == _chico].mean()), 0)}
# Los cuatro perfiles, por nombre (no por número)
_tasas = cancela.groupby(grupos).mean()
_cif["perfiles"] = {nombre[g]: {"n": int(perfiles.loc[g, "clientes"]), "edad": round(float(perfiles.loc[g, "edad"]), 1),
                                "usuarios": round(float(perfiles.loc[g, "usuarios"]), 2),
                                "consumo": round(float(perfiles.loc[g, "consumo_gb"]), 1),
                                "pymes": round(float(perfiles.loc[g, "pymes"]) * 100, 1),
                                "tasa": round(float(_tasas[g]) * 100, 1),
                                "edad_rango": [int(X["edad"][grupos == g].quantile(0.1)), int(X["edad"][grupos == g].quantile(0.9))],
                                "usuarios_rango": [int(X["usuarios"][grupos == g].quantile(0.1)), int(X["usuarios"][grupos == g].quantile(0.9))]}
                    for g in perfiles.index}
_cif["inercia_k4"] = round(float(km4.inertia_), 1)
_cif["iteraciones_k4"] = int(km4.n_iter_)
# n_init="auto": una sola corrida
_auto = KMeans(n_clusters=4, n_init="auto", random_state=2026).fit(Z)
_cif["n_init_auto"] = {"tamanos": sorted(map(int, np.bincount(_auto.labels_))), "inercia": round(float(_auto.inertia_), 1),
                       "tamanos_n10": sorted(map(int, np.bincount(grupos)))}
# Jerárquico
from scipy.cluster.hierarchy import fcluster
_f = fcluster(uniones, 4, "maxclust")
_cif["dendrograma_4"] = sorted(map(int, np.bincount(_f)[1:]), reverse=True)
_cif["encadena"] = {e: sorted(map(int, np.bincount(AgglomerativeClustering(n_clusters=4, linkage=e).fit_predict(Z))), reverse=True)
                    for e in ["ward", "single"]}
_ward = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Z)
_cif["ward_vs_kmeans_ari"] = round(float(adjusted_rand_score(_ward, grupos)), 3)
# Lunas
_ym = make_moons(n_samples=400, noise=0.08, random_state=2026)[1]
_cif["lunas"] = {"ari_kmeans": round(float(adjusted_rand_score(_ym, por_kmeans)), 3),
                 "ari_dbscan": round(float(adjusted_rand_score(_ym, por_dbscan)), 3),
                 "ruido_dbscan": int((por_dbscan == -1).sum()), "grupos_dbscan": int(por_dbscan.max() + 1)}
# DBSCAN en Yarumo
_d = {}
for _eps in (0.3, 0.5):
    _l = DBSCAN(eps=_eps, min_samples=20).fit_predict(Z)
    _r = _l == -1
    _d[str(_eps)] = {"grupos": int(_l.max() + 1), "ruido": int(_r.sum()),
                     "pymes_en_ruido": round(float(_pyme[_r].mean()) * 100, 1),
                     "pymes_que_son_ruido": round(float(_r[_pyme].mean()) * 100, 1),
                     "tamanos": sorted(map(int, np.bincount(_l[_l >= 0])), reverse=True),
                     "consumo_ruido": round(float(X["consumo_gb"][_r].mean()), 0),
                     "consumo_resto": round(float(X["consumo_gb"][~_r].mean()), 0),
                     "consumo_ruido_min": round(float(X["consumo_gb"][_r].min()), 0)}
    if _eps == 0.3:
        _ct = pd.crosstab(_l, clientes["usuarios"])
        _d["0.3"]["usuarios_por_grupo"] = {int(g): [int(u) for u in _ct.columns[_ct.loc[g] > 0]] for g in _ct.index if g >= 0}
    _l5 = DBSCAN(eps=_eps).fit_predict(Z)
    _d[str(_eps)]["min_samples_5"] = {"grupos": int(_l5.max() + 1), "ruido": int((_l5 == -1).sum())}
_cif["dbscan"] = _d
# Ejercicio 4: el grupo que se parte con plan_mbps
_ct4 = pd.crosstab(grupos, grupos5)   # por número: se traduce con nombre[g] abajo
_cif["ej4"] = {"tabla": {nombre[g]: [int(v) for v in _ct4.loc[g]] for g in _ct4.index},
               "plan_medio_nuevos": [round(float(clientes["plan_mbps"][grupos5 == j].mean()), 0) for j in range(5)],
               "edad_media_nuevos": [round(float(X["edad"][grupos5 == j].mean()), 1) for j in range(5)]}
_cif["segundos_precalculo"] = round(time.time() - _t0, 1)
print(json.dumps(_cif, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
