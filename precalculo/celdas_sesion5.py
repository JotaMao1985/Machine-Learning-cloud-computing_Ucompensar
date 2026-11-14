# Fuente ÚNICA del código de la Sesión 5. De aquí salen:
#   - el notebook para Colab                    (notebook.py genera 5)
#   - las salidas y las gráficas que muestran las diapositivas
#                                               (notebook.py ejecuta 5 → salidas_sesion5.json y
#                                                fuente/diapositivas/recursos/s5/<id>.svg)
#
# No es un script ejecutable: es el formato «percent» de Jupyter.
#   # %% [markdown]            celda de texto (líneas con «# »)
#   # %% <id> [banderas]       celda de código
# Banderas:
#   generador   en el notebook se antepone el código de descarga_datos.py, ayudas.py y
#               ayudas_sesion5.py (celda oculta); los datos se descargan (M3)
#   solucion    solución de un ejercicio: se ejecuta para verificar, NO va al notebook
#   plantilla   ejercicio por completar: va al notebook, NO se ejecuta al verificar
#   precalculo  cifras que las diapositivas citan en el texto: se ejecuta, NO va al notebook
#
# Todo es numpy y pandas (scikit-learn no trae aprendizaje por refuerzo, F-ML-73). Cada cálculo al
# azar lleva su semilla: las cifras son las mismas en cualquier equipo con las mismas versiones. Los
# valores de las políticas de Yarumo son EXACTOS (evaluación hacia atrás, sin simular): no dependen de
# una semilla. La celda [semillas] comprueba con otras semillas lo que el texto dice «en todas».

# %% [markdown]
# # Sesión 5 · Aprender por ensayo y error
# **Machine Learning y Seguridad en la Nube — Especialización en Big Data (UCompensar)**
#
# Este notebook acompaña las [diapositivas de la sesión](https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/diapositivas/sesion-5-aprender-por-ensayo-y-error.html).
# Cada celda lleva entre corchetes el mismo identificador que el código proyectado, así que puedes seguir
# la clase celda a celda. La portada del curso está en https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/.
#
# 1. **Archivo → Guardar una copia en Drive** para que tus cambios no se pierdan.
# 2. Ejecuta las celdas **en orden** con **Mayús + Enter**. Varias usan resultados de las anteriores.
#    Si Colab avisa que el notebook no es de Google, elige **Ejecutar de todos modos**.
# 3. Si Colab se reinicia o una celda falla al repetirla, usa **Entorno de ejecución → Ejecutar anteriores**.
#    Mientras tengan `...` sin reemplazar, las celdas de ejercicio dan error o un mensaje con ✗ (si el error
#    menciona `Ellipsis` o `ellipsis`, te quedó un `...`). Con **Entorno de ejecución → Ejecutar todas**, Colab se
#    detiene en el primer error (el ejercicio 2): complétalo, o sigue con **Mayús + Enter** celda por celda.
# 4. La primera celda **descarga** los datos del caso *Yarumo Fibra S.A.S.*, una empresa **ficticia** (sus
#    datos se generaron para el curso y no corresponden a personas ni a empresas reales), y comprueba que
#    el archivo sea idéntico al del curso. Si la descarga falla, la celda te dice cómo subirlo a mano.
# 5. Hoy todo es **numpy**, **pandas** y **matplotlib**: scikit-learn no trae aprendizaje por refuerzo. El notebook entero
#    corre en menos de un minuto, sin GPU. Cada cálculo al azar lleva su semilla, así que tus cifras deben
#    salir iguales a las de las diapositivas.
#
# En el código, los decimales van con punto (así los escribe Python); en el texto, con coma.

# %% datos generador
# Descarga datos/clientes.csv: los mismos 8000 clientes de las sesiones 1 a 4
ruta = descargar_datos("datos")

# %% reloj precalculo
import time
_t_inicio = time.time()   # para medir cuánto tarda el notebook del estudiante

# %% [markdown]
# ---
# # Parte 1 · Qué cambia
#
# En la sesión 2, cada cliente traía su **respuesta correcta** (si canceló o no) y el modelo aprendía a
# imitarla. En la sesión 3 no había respuesta: buscábamos grupos. Hoy, otra forma de aprender: un **agente**
# toma decisiones en un **entorno**, y nadie le dice cuál era la correcta. Solo recibe una **recompensa**,
# a veces mucho después, que depende de lo que hizo.
#
# - El **estado** es lo que el agente ve antes de decidir (dónde está, en qué mes va el cliente).
# - La **acción** es lo que elige; la **política** es su regla para elegir en cada estado.
# - La **recompensa** es un número que el entorno le devuelve después de cada acción.
# - Un **episodio** es una partida completa, de la salida al final.
# - El **retorno** es la suma de recompensas de un episodio. Con un **factor de descuento** γ (gamma), lo
#   lejano vale menos: $G = R_1 + \gamma R_2 + \gamma^2 R_3 + \dots$
#
# El entorno puede ser **azaroso**: la misma acción no siempre da lo mismo. Por eso se habla de lo que el
# agente **espera** ganar, el promedio de muchas partidas. Y hay un dilema: **explotar** lo que ya sabe o
# **explorar** algo nuevo que quizá sea mejor.
#
# ---
# # Parte 2 · Q-learning en una rejilla
#
# ## 1 · El entorno: una rejilla de 4 × 4
#
# El agente sale de la casilla de abajo a la izquierda (fila 3, columna 0) y quiere llegar a la meta, arriba a
# la derecha, que da **+10**. Hay dos pozos que dan **−10**. Llegar a la meta o caer en un pozo termina el
# episodio. Las demás casillas dan 0: **no hay costo por paso**, y lo que hace preferir el camino corto es γ.
# Las acciones son 0 (arriba), 1 (derecha), 2 (abajo) y 3 (izquierda); si choca con el borde, se queda donde
# está.

# %% rejilla
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILAS, COLUMNAS = 4, 4
INICIO, META = (3, 0), (0, 3)
POZOS = [(1, 1), (2, 3)]
MOVIMIENTOS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # ↑ → ↓ ←

def paso(s, a):
    """El entorno: desde la casilla s, con la acción a → (s2, r, fin)."""
    f = min(max(s[0] + MOVIMIENTOS[a][0], 0), FILAS - 1)
    c = min(max(s[1] + MOVIMIENTOS[a][1], 0), COLUMNAS - 1)
    s2 = (f, c)
    if s2 == META:
        return s2, 10, True       # llegó: +10 y termina
    if s2 in POZOS:
        return s2, -10, True      # cayó: −10 y termina
    return s2, 0, False           # sigue, sin recompensa

dibujar_rejilla(titulo="La rejilla")

# %% [markdown]
# ## 2 · Un agente al azar
#
# Antes de aprender nada, ¿qué pasa si el agente elige cada acción al azar? Lo probamos con 10 000 episodios,
# con un tope de 50 pasos por episodio.

# %% azar
rng = np.random.default_rng(2026)
final = {"pozo": 0, "meta": 0, "tope de 50 pasos": 0}
for episodio in range(10_000):
    s, fin = INICIO, False
    for t in range(50):
        s, r, fin = paso(s, int(rng.integers(4)))  # una acción al azar
        if fin:
            break
    if not fin:
        final["tope de 50 pasos"] += 1
    else:
        final["meta" if s == META else "pozo"] += 1
for como, cuantos in final.items():
    print(f"{como}: {cuantos / 10_000:.1%}")

# %% [markdown]
# ## 3 · La tabla Q y la regla de Q-learning
#
# El agente guarda una **tabla Q**: para cada casilla y cada acción, **lo que espera ganar si hace esto aquí**,
# con lo que vale menos lo lejano. Al principio todo vale 0. Después de cada paso, mueve ese valor un poquito
# hacia lo que acaba de ver (Sutton y Barto, ecuación 6.8):
#
# $$Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\big]$$
#
# - $r + \gamma \max Q(s', \cdot)$ es el **objetivo**: la recompensa que recibió más lo mejor que espera ganar
#   desde la casilla nueva, descontado. Si el paso terminó el episodio, ya no hay después: ese término vale 0.
# - α (alfa) es la **tasa de aprendizaje**, como en la sesión 4: qué fracción del camino hacia el objetivo
#   avanza en cada paso.
#
# Dos pasos a mano, con α = 0,1, γ = 0,9 y la tabla en ceros: en un episodio, el agente va a la derecha desde
# (0, 2) y llega a la meta; en un episodio posterior, va a la derecha desde (0, 1) y llega a (0, 2).

# %% a_mano
alfa, gamma = 0.1, 0.9
# 1. Desde (0, 2) va a la meta: r = 10 y termina (no hay después)
q_meta = 0 + alfa * (10 + 0 - 0)
# 2. Desde (0, 1) va a (0, 2): r = 0,
#    y lo mejor que espera ganar en (0, 2) ya es q_meta
q_antes = 0 + alfa * (0 + gamma * q_meta - 0)
print(f"Q[(0, 2), derecha] = {q_meta:.2f}")
print(f"Q[(0, 1), derecha] = {q_antes:.2f}")

# %% [markdown]
# **La recompensa viaja hacia atrás.** La casilla (0, 1) nunca tocó la meta, pero ya sabe que ir a la derecha
# vale algo, porque desde (0, 2) se llega a la meta. Con más episodios, el valor sigue retrocediendo, casilla a casilla,
# hasta la salida.
#
# ---
# ## Ejercicio 1 · La actualización de Q
#
# Completa la función `actualiza`: recibe la tabla `Q`, la casilla `s`, la acción `a`, la recompensa `r`, la
# casilla nueva `s2` y si el episodio terminó (`fin`), y **devuelve el valor nuevo de `Q[s][a]`** según la regla
# de arriba. No cambies la primera línea (la de `def`). El verificador la prueba con casos que no son de la
# rejilla, y si algo no cuadra te dice qué parte de la regla revisar.

# %% ej1 plantilla
def actualiza(Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9):
    """Devuelve el valor nuevo de Q[s][a] (no hace falta cambiar Q)."""
    lo_mejor_despues = 0 if fin else Q[s2].max()  # si terminó, no hay más
    objetivo = ...  # la recompensa más lo mejor de después, descontado
    return ...      # el valor actual avanza una fracción alfa hacia él

revisar_actualiza(actualiza)

# %% ej1_solucion solucion
def actualiza(Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9):
    """Devuelve el valor nuevo de Q[s][a] (no hace falta cambiar Q)."""
    lo_mejor_despues = 0 if fin else Q[s2].max()
    objetivo = r + gamma * lo_mejor_despues
    return Q[s][a] + alfa * (objetivo - Q[s][a])

revisar_actualiza(actualiza)

# %% [markdown]
# ## 4 · Entrenar: explorar un poco, explotar casi siempre
#
# En cada paso, el agente sigue la **regla ε-codiciosa** (*ε-greedy*): con probabilidad ε elige una acción al
# azar (**explora**); el resto de las veces, la de mayor valor en su tabla (**explota**). Si varias empatan,
# `np.argmax` elige la primera. Además cuenta cuántas veces pasó por cada casilla (`visitas`).

# %% entrenar
def entrenar_rejilla(eps=0.1, episodios=500, alfa=0.1, gamma=0.9,
                     semilla=2026):
    rng = np.random.default_rng(semilla)
    Q = np.zeros((FILAS, COLUMNAS, 4))   # al principio, todo vale 0
    visitas = np.zeros((FILAS, COLUMNAS), dtype=int)
    for episodio in range(episodios):
        s = INICIO
        for t in range(50):                  # tope de 50 pasos
            visitas[s] += 1
            if rng.random() < eps:
                a = int(rng.integers(4))     # explora: al azar
            else:
                a = int(np.argmax(Q[s]))     # explota: la mejor
            s2, r, fin = paso(s, a)
            mejor_despues = 0 if fin else Q[s2].max()
            Q[s][a] += alfa * (r + gamma * mejor_despues - Q[s][a])
            s = s2
            if fin:
                break
    return Q, visitas

# %% voraz
def camino_voraz(Q):
    """Desde la salida, siempre la mejor acción (máx. 50 pasos)."""
    s, camino = INICIO, [INICIO]
    for t in range(50):
        s, r, fin = paso(s, int(np.argmax(Q[s])))
        camino.append(s)
        if fin:
            break
    return camino

def llega_a_la_meta(Q):
    return camino_voraz(Q)[-1] == META

# %% aprende
Q, visitas = entrenar_rejilla(eps=0.1)
camino = camino_voraz(Q)
print(f"¿Llega? {llega_a_la_meta(Q)}, en {len(camino) - 1} pasos")
print(f"Lo que espera ganar desde la salida: {Q[INICIO].max():.2f}")
print(f"La meta a 6 pasos: 10 · 0.9^5 = {10 * 0.9 ** 5:.2f}")

# %% [markdown]
# ¿Por qué 0,9⁵? La recompensa llega en el sexto paso, y cada paso que la separa de la salida la multiplica por γ:
# la primera acción vale $10\gamma^5$. Si la ruta fuera de 8 pasos, valdría $10\gamma^7$, menos: la ruta corta vale
# más. Prueba `entrenar_rejilla(gamma=1.0)`: ¿cuánto vale ahora la salida? Sin γ, todas las rutas que llegan valen
# lo mismo.
#
# El dibujo muestra, en cada casilla libre, la acción de mayor valor (la flecha) y cuántas veces pasó el agente
# por ella en los 500 episodios. Un punto gris quiere decir que todas sus acciones valen 0: allí no aprendió nada.
# Q-learning aprende **fuera de política** (*off-policy*): sus valores apuntan a la mejor política aunque el agente
# explore, porque la regla usa el máximo y no la jugada al azar que a veces hace.

# %% flechas
dibujar_rejilla(Q, visitas, "Lo aprendido con ε = 0,1 (500 episodios)")

# %% [markdown]
# Mira las casillas con pocas visitas: con tan pocas pruebas, su flecha no es confiable. El agente aprende bien
# lo que recorre a menudo.
#
# ---
# ## Ejercicio 2 · Sin explorar
#
# Entrena dos agentes con la semilla de siempre: uno **sin explorar** (ε = 0) y otro con ε = 0,1. Antes de
# ejecutar, **predice** cuál llega a la meta. Después, cuenta cuántas casillas distintas visitó el que no explora
# (`casillas_eps0`), mira cuáles son y elige la explicación (la letra, entre comillas, en `razon`):
#
# - **A.** Sin exploración, la tasa de aprendizaje α deja de actuar: la tabla Q se actualiza, pero en pasos tan
#   pequeños que no alcanzan a cambiar la decisión.
# - **B.** Con γ = 0,9, una meta a seis pasos queda tan descontada que, para el agente, ir hasta ella vale
#   menos que quedarse donde está.
# - **C.** La tabla empieza en ceros: entre acciones empatadas elige siempre la misma y, como nunca recibe una
#   recompensa, nada lo saca de ese camino.
# - **D.** 500 episodios no alcanzan para un agente que no explora: con unos miles más, también encontraría la
#   ruta hasta la meta.

# %% ej2 plantilla
Q_0, visitas_0 = entrenar_rejilla(eps=...)     # no explora nunca
Q_01, visitas_01 = entrenar_rejilla(eps=...)   # explora el 10 %
llega_eps0 = llega_a_la_meta(Q_0)
llega_eps01 = ...                              # ¿y el que explora?
dibujar_rejilla(Q_0, visitas_0, "Sin explorar (ε = 0)")
casillas_eps0 = ...  # cuántas casillas distintas visitó (pista: visitas_0 > 0)
razon = "..."        # la letra de la explicación, entre comillas
revisar_exploracion(llega_eps0=llega_eps0, llega_eps01=llega_eps01,
                    casillas_eps0=casillas_eps0, razon=razon)

# %% ej2_solucion solucion
Q_0, visitas_0 = entrenar_rejilla(eps=0.0)
Q_01, visitas_01 = entrenar_rejilla(eps=0.1)
llega_eps0 = llega_a_la_meta(Q_0)
llega_eps01 = llega_a_la_meta(Q_01)
dibujar_rejilla(Q_0, visitas_0, "Sin explorar (ε = 0)")
casillas_eps0 = int((visitas_0 > 0).sum())
razon = "C"
revisar_exploracion(llega_eps0=llega_eps0, llega_eps01=llega_eps01,
                    casillas_eps0=casillas_eps0, razon=razon)

# %% [markdown]
# ---
# # Parte 3 · La retención de Yarumo
#
# En la sesión 1 quedó una pregunta para hoy: **¿qué oferta de retención hacer, y cuándo?** Lo planteamos como
# un problema de aprendizaje por refuerzo, con un **simulador**: un entorno de mentira con los supuestos a la vista.
#
# - **El episodio** es un cliente nuevo durante 24 meses. **El estado** es el mes del cliente.
# - **Las acciones**, cada mes: 0 **nada**, 1 **llamar** o 2 **oferta de retención**.
# - **La recompensa**: si el cliente sigue ese mes, Yarumo cobra la factura; la acción cuesta lo que cuesta,
#   siga o no. Es facturación, no ganancia, y no se usa γ (equivale a γ = 1): el horizonte es de 24 meses.
#
# ## 5 · El riesgo de cancelar sale de los datos
#
# Para cada tramo de antigüedad calculamos la tasa de cancelación a 3 meses (julio a septiembre) de `clientes.csv`. Hasta
# el mes 12 tomamos a los clientes **con** el descuento de bienvenida, que vence entre los meses 10 y 12; del 13
# al 24, a los que **ya no** lo tienen. La tasa a 3 meses se pasa a un riesgo **mensual** suponiendo que es
# parejo en los tres meses: $1 - (1 - p)^{1/3}$.

# %% riesgo
clientes = pd.read_csv("datos/clientes.csv")
ant = clientes["antiguedad_meses"]
con_bienvenida = (ant <= 12) & (clientes["descuento"] == "Sí")
ya_sin = ant.between(13, 24) & (clientes["descuento"] == "No")
grupo = clientes[con_bienvenida | ya_sin]
tramo_ant = pd.cut(grupo["antiguedad_meses"], [0, 9, 12, 24],
                   labels=["meses 1-9", "meses 10-12", "meses 13-24"])
cancelo = (grupo["cancelo"] == "Sí")
tabla_riesgo = cancelo.groupby(tramo_ant, observed=True).agg(
    clientes="size", a_3_meses="mean")
tabla_riesgo["al_mes"] = 1 - (1 - tabla_riesgo["a_3_meses"]) ** (1 / 3)
RIESGO = tabla_riesgo["al_mes"].to_numpy()
print(tabla_riesgo.round(3))

# %% [markdown]
# ## 6 · El simulador: lo que sale de los datos y lo que es inventado
#
# La factura es la mediana de `clientes.csv`, en miles de pesos (ficticios). **El efecto y el costo de cada
# acción son INVENTADOS**: no hay en los datos ningún experimento que los mida. Supondremos que llamar reduce el
# riesgo del mes en una cuarta parte (lo multiplica por 0,75) y cuesta 20 mil pesos, y que la oferta de retención
# lo reduce en un 60 % (lo multiplica por 0,4) y cuesta 60 mil.
#
# `valor_politica` calcula **exacto**, sin simular, lo que deja en promedio un cliente con una política, en miles
# de pesos. Va del mes 24 hacia el 1: cada mes se paga la acción y, si el cliente no cancela, se cobra la factura
# y se suma lo que vale el resto de los meses.

# %% entorno
FACTURA = round(clientes["factura_mensual"].median() / 1000)  # miles
ACCIONES = ["nada", "llamar", "oferta de retención"]
BAJA_RIESGO = [0, 0.25, 0.60]   # INVENTADO: cuánto baja el riesgo del mes
COSTO = [0, 20, 60]             # INVENTADO: miles de pesos por acción
MESES = 24

def tramo(mes):
    return 0 if mes <= 9 else (1 if mes <= 12 else 2)

def riesgo(mes, accion):
    return RIESGO[tramo(mes)] * (1 - BAJA_RIESGO[accion])

def valor_politica(politica):
    """Lo que deja en promedio un cliente en 24 meses, exacto."""
    valor = 0.0
    for mes in range(MESES, 0, -1):          # de atrás hacia adelante
        a = politica(mes)
        valor = -COSTO[a] + (1 - riesgo(mes, a)) * (FACTURA + valor)
    return valor

print(f"Factura mediana: {FACTURA} mil pesos al mes")

# %% politicas
politicas = {
    "No hacer nada": lambda mes: 0,
    "Llamar siempre": lambda mes: 1,
    "Oferta siempre": lambda mes: 2,
    "Oferta solo en los meses 10–12": lambda mes: 2 if 10 <= mes <= 12
                                      else 0,
}
for nombre, politica in politicas.items():
    print(f"{nombre:31} {valor_politica(politica):6.1f} mil por cliente")

# %% [markdown]
# ## 7 · La mejor política posible, porque aquí conocemos el entorno
#
# Como el simulador es nuestro, conocemos el riesgo de cada mes con cada acción, y podemos calcular la mejor
# política de atrás hacia adelante: en el mes 24, la mejor acción; en el 23, la mejor sabiendo lo que vale el
# 24; y así hasta el mes 1. **En la vida real nadie conoce esas probabilidades**: por eso el refuerzo aprende
# probando. Este cálculo solo sirve para medir qué tan cerca llega el agente.

# %% mejor
def mejor_politica():
    valor, mejor = 0.0, {}
    for mes in range(MESES, 0, -1):
        opciones = [-COSTO[a] + (1 - riesgo(mes, a)) * (FACTURA + valor)
                    for a in range(3)]
        mejor[mes] = int(np.argmax(opciones))
        valor = max(opciones)
    return mejor, valor

mejor, valor_mejor = mejor_politica()
print("Mes a mes (0 nada, 1 llamar, 2 oferta):")
print("".join(str(mejor[mes]) for mes in range(1, MESES + 1)))
print(f"La mejor posible: {valor_mejor:.1f} mil por cliente")

# %% [markdown]
# ## 8 · Q-learning con 20 000 clientes simulados
#
# El agente **no** ve el riesgo: solo prueba acciones con clientes simulados y mira qué pasa. Es la regla de la
# rejilla, con tres ajustes: sin γ (equivale a γ = 1); explora mucho al principio (ε baja en línea recta desde 1
# y llega a 0,3 en el cliente 7000) y después no baja más, y da pasos α cada vez más cortos en cada mes y acción, porque aquí el
# resultado es azaroso y cada cliente simulado trae ruido.

# %% q_yarumo
def q_learning_yarumo(clientes_sim=20_000, semilla=2026):
    rng = np.random.default_rng(semilla)
    Q = np.zeros((MESES + 2, 3))  # fila = mes, columna = acción
    N = np.zeros((MESES + 2, 3))  # veces que probó cada una
    for k in range(clientes_sim):
        eps = max(0.3, 1 - k / 10_000)        # explora menos con el tiempo
        for mes in range(1, MESES + 1):
            if rng.random() < eps:
                a = int(rng.integers(3))
            else:
                a = int(np.argmax(Q[mes]))
            cancela = rng.random() < riesgo(mes, a)  # el entorno decide
            r = -COSTO[a] + (0 if cancela else FACTURA)
            fin = cancela or mes == MESES
            N[mes, a] += 1
            alfa = max(0.001, 1 / N[mes, a] ** 0.8)  # pasos más cortos
            despues = 0 if fin else Q[mes + 1].max()
            Q[mes, a] += alfa * (r + despues - Q[mes, a])
            if fin:
                break
    return Q

# %% aprendida
Q_y = q_learning_yarumo()
aprendida = {mes: int(np.argmax(Q_y[mes])) for mes in range(1, MESES + 1)}
print("".join(str(aprendida[mes]) for mes in range(1, MESES + 1)))
valor_aprendida = valor_politica(lambda mes: aprendida[mes])
print(f"Lo aprendido vale {valor_aprendida:.1f} mil por cliente")

# %% planes
regla = [2 if 10 <= mes <= 12 else 0 for mes in range(1, MESES + 1)]
dibujar_planes({
    "Oferta en 10–12": regla,
    "Q-learning": [aprendida[mes] for mes in range(1, MESES + 1)],
    "La mejor posible": [mejor[mes] for mes in range(1, MESES + 1)],
})

# %% [markdown]
# Cuatro cosas, con honestidad:
#
# 1. **El agente redescubre lo que el simulador supone.** El valor está en el método: en la vida real, el
#    simulador se estima con datos (por ejemplo, de campañas pasadas) y se valida.
# 2. **Solo lo de los meses 10 a 12 resiste el ruido**, y solo si el efecto inventado de la oferta es cierto.
#    En los demás meses, cada mes las acciones valen casi lo mismo y el agente no distingue entre ellas, aunque,
#    sumadas, esas pequeñas diferencias son buena parte de lo que separa a la regla de la mejor posible.
# 3. **«La mejor posible» se calculó porque conocemos el entorno.** El refuerzo sirve cuando no se conoce.
# 4. **Hicieron falta 20 000 clientes simulados.** No se aprende experimentando así con clientes reales: las
#    acciones de exploración rara vez están libres de consecuencias (Dulac-Arnold et al., 2021). Y si una política
#    actúa sola, hay que decidir quién la aprueba y qué se registra de cada decisión.
#
# ---
# ## Ejercicio 3 · Tu política de retención
#
# Escribe tu propia política: una función que recibe el mes (de 1 a 24) y devuelve la acción (0 nada, 1 llamar,
# 2 oferta de retención). Calcula su valor con `valor_politica` (no lo escribas a mano) y compárala con la regla
# de los meses 10 a 12 y con la mejor posible. La regla de los meses 10 a 12 tiene tres tramos: puedes agregar o
# quitar los que quieras.

# %% ej3 plantilla
def mi_politica(mes):
    """Del mes 1 al 24: 0 nada, 1 llamar, 2 oferta de retención."""
    if mes <= ...:
        return ...
    if mes <= ...:       # puedes agregar o quitar tramos
        return ...
    return ...

valor = ...    # calcúlalo con valor_politica
revisar_politica(mi_politica, valor=valor)

# %% ej3_solucion solucion
def mi_politica(mes):
    """Del mes 1 al 24: 0 nada, 1 llamar, 2 oferta de retención."""
    if 10 <= mes <= 12:
        return 2      # la oferta, cuando vence el descuento de bienvenida
    if mes >= 13:
        return 1      # después, llamar
    return 0

valor = valor_politica(mi_politica)
revisar_politica(mi_politica, valor=valor)

# %% [markdown]
# ---
# # Parte 4 · *Policy gradients*: ajustar las probabilidades
#
# Q-learning estima **cuánto vale** cada acción y elige la mejor. Los métodos de **gradiente de política**
# (*policy gradients*) no estiman valores: la política es directamente una probabilidad para cada acción, como
# la salida **softmax** de la red de dígitos de la sesión 4. Después de cada cliente simulado, **sube la
# probabilidad de lo que salió mejor que un valor de referencia y baja la de lo que salió peor**. Es la idea de
# REINFORCE (Williams, 1992; Sutton y Barto, ecuaciones 13.8 y 13.11).
#
# Aquí la política tiene tres preferencias por tramo (meses 1–9, 10–12 y 13–24), una por acción. El valor de
# referencia es el promedio de lo que se ha ganado desde cada mes; no depende de la acción elegida. Corre la
# celda y **mira**: no hay ejercicio.

# %% reinforce
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def reinforce(semilla, clientes_sim=12_000, alfa=0.005):
    rng = np.random.default_rng(semilla)
    pref = np.zeros((3, 3))           # preferencias: tramo × acción
    referencia = np.zeros(MESES + 1)  # lo que se suele ganar desde cada mes
    historia = []
    for k in range(clientes_sim):
        tramos, acciones, recompensas = [], [], []
        for mes in range(1, MESES + 1):            # un cliente simulado
            pi = softmax(pref[tramo(mes)])
            a = int(rng.choice(3, p=pi))   # elige con probabilidades
            tramos.append(tramo(mes))
            acciones.append(a)
            if rng.random() < riesgo(mes, a):
                recompensas.append(-COSTO[a])
                break
            recompensas.append(FACTURA - COSTO[a])
        # lo que ganó desde cada mes hasta el final, en facturas
        G = np.cumsum(recompensas[::-1])[::-1] / FACTURA
        for t, (k_tramo, a) in enumerate(zip(tramos, acciones)):
            ventaja = G[t] - referencia[t + 1]  # ¿mejor que lo usual?
            referencia[t + 1] += 0.05 * (G[t] - referencia[t + 1])
            pi = softmax(pref[k_tramo])
            sube = -pi
            sube[a] += 1            # la elegida sube; las otras bajan
            pref[k_tramo] += alfa * ventaja * sube
        historia.append([softmax(p) for p in pref])
    return pref, np.array(historia)

# %% reinforce_semillas
semillas = [2026, 1, 2, 3, 4]
fig, ejes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
for semilla in semillas:
    pref, historia = reinforce(semilla)
    x = np.arange(0, len(historia), 40)          # un punto cada 40 clientes
    ejes[0].plot(x, historia[x, 1, 2], label=f"semilla {semilla}")
    ejes[1].plot(x, historia[x, 0, 1])
    planes = [ACCIONES[int(np.argmax(p))] for p in pref]
    print(f"Semilla {semilla}: 1-9 {planes[0]}, 10-12 {planes[1]}, "
          f"13-24 {planes[2]}")
ejes[0].set_title("Meses 10–12: P(oferta)")
ejes[1].set_title("Meses 1–9: P(llamar)")
for eje in ejes:
    eje.set_xlabel("Clientes simulados")
ejes[0].legend(fontsize=10)
plt.show()

# %% [markdown]
# En los meses 10 a 12 (y, en estas cinco, también en los 13 a 24) todas las semillas aprenden lo mismo. En los
# meses 1 a 9, la diferencia entre las acciones es pequeña frente al ruido de cada cliente simulado, y lo que se
# aprende cambia de una semilla a otra: estos métodos pueden tener **varianza alta**. El valor de referencia la
# reduce, pero no la elimina.
#
# Un método de esta familia, PPO (Schulman et al., 2017), se usó para ajustar InstructGPT con las preferencias
# de personas (Ouyang et al., 2022): allí la recompensa la da un modelo entrenado con comparaciones hechas por personas.
#
# ---
# ### Tus respuestas
#
# *Doble clic para escribir aquí.*
#
# 1. Ejercicio 1: en la regla de Q-learning, ¿qué término hace que la recompensa de la meta llegue, con los
#    episodios, hasta la salida? …
# 2. Ejercicio 2: el agente que no explora visitó … casillas distintas. ¿Por qué no llega? ¿Se parece a una
#    empresa que solo repite lo que ya le funcionó? …
# 3. Ejercicio 3: mi política fue … y vale … mil pesos por cliente; la regla de los meses 10–12 vale … y la mejor
#    posible, … ¿Por qué la tuya gana o pierde? ¿Qué supuesto del simulador, si cambiara, cambiaría tu
#    respuesta? …
# 4. Gradiente de política: ¿en qué tramos coinciden las cinco semillas y en cuál no? ¿Por qué crees que pasa? …
# 5. Si Yarumo dejara que una política así decidiera sola, ¿quién debería aprobarla y qué registrarías de cada
#    decisión? …
#
# ---
# ## La respuesta a la gerencia
#
# **¿Qué oferta de retención hacer, y cuándo?** Bajo los supuestos de este simulador, la oferta vale la pena
# cuando vence el descuento de bienvenida (meses 10 a 12), que es cuando más cancelan; en los demás meses, la
# diferencia entre llamar y no hacer nada es pequeña. El agente lo aprendió con 20 000 clientes **simulados**: el
# siguiente paso no es probarlo con clientes reales, sino estimar el efecto de cada acción con datos de campañas
# y validar el simulador. Si una política decide sola, alguien la aprueba y cada decisión queda registrada.

# %% cifras precalculo
# Cifras que las diapositivas y sus notas citan (no va al notebook)
import json
_cif = {"tiempo_notebook_s": round(time.time() - _t_inicio, 1)}
_cif["azar"] = {k: round(v / 10_000, 4) for k, v in final.items()}
_cif["a_mano"] = {"q_meta": q_meta, "q_antes": round(q_antes, 4)}
_cif["rejilla"] = {"llega": bool(llega_a_la_meta(Q)), "pasos": len(camino) - 1,
                   "q_inicio": round(float(Q[INICIO].max()), 4), "objetivo": round(10 * 0.9 ** 5, 4),
                   "camino": [list(map(int, s)) for s in camino],
                   "visitas": visitas.tolist(),
                   "libres_min_visitas": min(((int(visitas[f, c]), (f, c)) for f in range(4) for c in range(4)
                                              if (f, c) != META and (f, c) not in POZOS)),
                   "flechas": {f"{f},{c}": int(np.argmax(Q[f, c])) for f in range(4) for c in range(4)}}
_cif["ej2"] = {"llega_eps0": bool(llega_eps0), "llega_eps01": bool(llega_eps01),
               "visitadas_eps0": int((visitas_0 > 0).sum()),
               "casillas_eps0": [[f, c] for f in range(4) for c in range(4) if visitas_0[f, c] > 0]}
_cif["riesgo"] = {str(k): {"clientes": int(v["clientes"]), "a_3_meses": round(float(v["a_3_meses"]), 4),
                           "al_mes": round(float(v["al_mes"]), 4)} for k, v in tabla_riesgo.iterrows()}
_cif["factura"] = FACTURA
_cif["rangos"] = {"edad": [int(clientes["edad"].min()), int(clientes["edad"].max())],
                  "factura": [int(clientes["factura_mensual"].min()), int(clientes["factura_mensual"].max())]}
_cif["politicas"] = {n: round(valor_politica(p), 2) for n, p in politicas.items()}
_cif["mejor"] = {"plan": "".join(str(mejor[m]) for m in range(1, MESES + 1)), "valor": round(valor_mejor, 2)}
_cif["q_yarumo"] = {"plan": "".join(str(aprendida[m]) for m in range(1, MESES + 1)),
                    "valor": round(valor_aprendida, 2),
                    "acierta_10_12": all(aprendida[m] == mejor[m] for m in (10, 11, 12)),
                    "aciertos_por_tramo": {t: sum(aprendida[m] == mejor[m] for m in r) for t, r in
                                           {"1-9": range(1, 10), "10-12": range(10, 13), "13-24": range(13, 25)}.items()}}
_cif["ej3"] = {"valor": round(valor, 2)}
# Sensibilidad: ¿desde qué efecto de la oferta la mejor política la usa en algún mes?
_baja_original = BAJA_RIESGO[2]
_cif["sensibilidad_oferta"] = {}
for _b in [round(0.20 + 0.01 * i, 2) for i in range(41)]:
    BAJA_RIESGO[2] = _b
    _m, _vm = mejor_politica()
    _cif["sensibilidad_oferta"][str(_b)] = {"plan": "".join(str(_m[m]) for m in range(1, MESES + 1)), "valor": round(_vm, 2)}
BAJA_RIESGO[2] = _baja_original
_cif["sensibilidad_oferta_umbral"] = min(float(b) for b, v in _cif["sensibilidad_oferta"].items() if "2" in v["plan"])
# Cuánto cambia el valor con el mes en que se hace la oferta: diferencia de llamar frente a nada en 1-9
_cif["diferencias_mejor"] = {}
_v = 0.0
for _m in range(MESES, 0, -1):
    _ops = [-COSTO[a] + (1 - riesgo(_m, a)) * (FACTURA + _v) for a in range(3)]
    _cif["diferencias_mejor"][_m] = [round(o, 2) for o in _ops]
    _v = max(_ops)
print(json.dumps(_cif, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

# %% semillas precalculo
# Lo que el texto dice «en todas las semillas probadas», comprobado aquí con otras semillas.
_sem = {}
# Rejilla: ¿llega con ε = 0,1? ¿y con ε = 0? (50 semillas, como el diseño)
_l01 = [llega_a_la_meta(entrenar_rejilla(eps=0.1, semilla=s)[0]) for s in range(50)]
_l0 = [llega_a_la_meta(entrenar_rejilla(eps=0.0, semilla=s)[0]) for s in range(50)]
_q0 = [float(entrenar_rejilla(eps=0.1, semilla=s)[0][INICIO].max()) for s in range(50)]
_sem["rejilla_50"] = {"llega_eps01": int(sum(_l01)), "llega_eps0": int(sum(_l0)),
                      "q_inicio_min": round(min(_q0), 3), "q_inicio_max": round(max(_q0), 3)}
# Yarumo: Q-learning con 30 semillas, valor exacto de lo aprendido
_vals, _acier = [], {"1-9": 0, "10-12": 0, "13-24": 0}
for s in range(30):
    _Qs = q_learning_yarumo(semilla=s)
    _p = {m: int(np.argmax(_Qs[m])) for m in range(1, MESES + 1)}
    _vals.append(valor_politica(lambda m: _p[m]))
    for t, r in {"1-9": range(1, 10), "10-12": range(10, 13), "13-24": range(13, 25)}.items():
        _acier[t] += sum(_p[m] == mejor[m] for m in r) / len(r)
_regla = valor_politica(politicas["Oferta solo en los meses 10–12"])
_sem["q_yarumo_30"] = {"media": round(float(np.mean(_vals)), 2), "de": round(float(np.std(_vals)), 2),
                       "min": round(min(_vals), 2), "max": round(max(_vals), 2),
                       "sobre_regla": int(sum(v > _regla for v in _vals)),
                       "acierto_medio": {t: round(v / 30, 3) for t, v in _acier.items()}}
# REINFORCE con 40 semillas: lo más probable al final en cada tramo
from collections import Counter
_rf = Counter()
_tramos = [Counter(), Counter(), Counter()]
for s in range(40):
    _pref, _ = reinforce(s)
    _plan = tuple(ACCIONES[int(np.argmax(p))] for p in _pref)
    _rf[_plan] += 1
    for i in range(3):
        _tramos[i][_plan[i]] += 1
_sem["reinforce_40"] = {"planes": {" / ".join(k): v for k, v in _rf.most_common()},
                        "por_tramo": [dict(c) for c in _tramos]}
_sem["tiempo_total_s"] = round(time.time() - _t_inicio, 1)
print(json.dumps(_sem, ensure_ascii=False))

# %% verificadores precalculo
# Los verificadores con los errores típicos: cada uno debe dar su pista, sin caerse.
import numpy as _np
_errores = {
    "sin r": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + alfa * ((0 if fin else gamma * Q[s2].max()) - Q[s][a]),
    "sin gamma": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + alfa * (r + (0 if fin else Q[s2].max()) - Q[s][a]),
    "sin max": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + alfa * (r - Q[s][a]),
    "sin alfa": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: r + (0 if fin else gamma * Q[s2].max()),
    "sin restar": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + alfa * (r + (0 if fin else gamma * Q[s2].max())),
    "a mano": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + 0.1 * (r + (0 if fin else 0.9 * Q[s2].max()) - Q[s][a]),
    "sin fin": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + alfa * (r + gamma * Q[s2].max() - Q[s][a]),
    "None": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: None,
    "error": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + ... ,
    "otra firma": lambda Q, s, a: 0,
    "orden contrario": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + alfa * (Q[s][a] - (r + (0 if fin else gamma * Q[s2].max()))),
    "Q[s2][a]": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s][a] + alfa * (r + (0 if fin else gamma * Q[s2][a]) - Q[s][a]),
    "Q[s].max()": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s].max() + alfa * (r + (0 if fin else gamma * Q[s2].max()) - Q[s].max()),
    "Q[s, a]": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: Q[s, a] + alfa * (r + (0 if fin else gamma * Q[s2].max()) - Q[s, a]),
    "float32": lambda Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9: _np.float32(Q[s][a] + alfa * (r + (0 if fin else gamma * Q[s2].max()) - Q[s][a])),
}
for _n, _f in _errores.items():
    print(f"--- {_n}")
    revisar_actualiza(_f)
print("--- sin paréntesis no / un número")
revisar_actualiza(3.0)
print("--- ej2: True en texto, letra minúscula")
revisar_exploracion(llega_eps0="False", llega_eps01=True, casillas_eps0=int((visitas_0 > 0).sum()), razon="c")
print("--- ej2: cruzados, letra D")
revisar_exploracion(llega_eps0=True, llega_eps01=False, casillas_eps0=int(visitas_0.sum()), razon="D")
print("--- ej2: posicional")
revisar_exploracion(True, False, 4, "C")
print("--- ej2: casillas como tabla, sin casillas")
revisar_exploracion(llega_eps0=False, llega_eps01=True, casillas_eps0=visitas_0 > 0, razon="C")
revisar_exploracion(llega_eps0=False, llega_eps01=True, razon="C")
print("--- ej3: texto, lista corta, None, valor a mano, valor de otra, lista de 24")
revisar_politica(lambda mes: "llamar", valor=900)
revisar_politica([1, 2, 0], valor=900)
revisar_politica(lambda mes: None, valor=900)
revisar_politica(lambda mes: 1, valor=900)
revisar_politica(lambda mes: 1, valor=round(valor_politica(lambda mes: 0), 1))
revisar_politica([1] * 9 + [2] * 3 + [0] * 12, valor=round(valor_mejor))
revisar_politica(lambda mes: 1.0, valor=1)
revisar_politica(..., valor=...)
revisar_politica(lambda mes: 0, valor=[900])
revisar_politica(_np.array([1] * 24, dtype=_np.int64), valor=valor_politica(lambda mes: 1))
