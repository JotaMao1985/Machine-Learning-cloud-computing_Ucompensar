"""Sesión 5: estilo de las gráficas del curso y los verificadores de la práctica.

Van dentro de la celda oculta del notebook, junto con descarga_datos.py y ayudas.py. Cada
verificador recibe lo que el estudiante hizo, dice qué está bien y, en lo demás, da una pista sin
la respuesta. Nunca se cae: un «...» sin reemplazar, un error dentro de la función del estudiante o
un tipo inesperado se avisa.

Los verificadores NO usan las funciones del notebook (el estudiante puede haberlas cambiado):
tienen aquí su propia copia de la rejilla y del simulador de Yarumo, con los mismos datos:

- ejercicio 1: prueban la función `actualiza` del estudiante con casos fijos en los que cada error
  típico da un valor distinto (sin la recompensa, sin descontar, sin lo mejor del estado siguiente,
  sin alfa, sin restar el valor actual, solo el cambio, alfa y gamma escritos a mano, sin tratar el
  final);
- ejercicio 2: entrenan aquí los dos agentes de la rejilla (ε = 0 y ε = 0,1, semilla 2026, 500
  episodios), comparan lo que el estudiante dice que pasó y revisan la letra de la explicación;
- ejercicio 3: llaman a la política del estudiante para los 24 meses y recalculan su valor exacto.
"""

_FILAS, _COLUMNAS = 4, 4
_INICIO, _META, _POZOS = (3, 0), (0, 3), ((1, 1), (2, 3))
_MOV = ((-1, 0), (0, 1), (1, 0), (0, -1))   # arriba, derecha, abajo, izquierda
_RAZON_BIEN = "C"


def _estilo_graficas():
    """Colores del curso y letras que se leen en Colab y proyectadas."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=["#8A2E9E", "#F46201", "#566B7F", "#0E7490", "#CA8A04"]),
        "figure.figsize": (6.4, 3.6),
        "figure.constrained_layout.use": True,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#94A3B8",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": "#1E293B",
    })


_estilo_graficas()

_CACHE = {}


def _seguro(verificador):
    """Red de seguridad: si algo imprevisto falla al revisar, se avisa sin traceback."""
    import functools

    @functools.wraps(verificador)
    def envoltura(*args, **kwargs):
        try:
            return verificador(*args, **kwargs)
        except Exception as error:   # noqa: BLE001 — el estudiante nunca ve un traceback
            print(f"✗  No pude revisar lo que entregaste ({type(error).__name__}). Revisa lo que pasaste al "
                  "verificador y vuelve a ejecutar la celda")
    return envoltura


def _coma(x, decimales=2):
    """Un número con coma decimal, para los mensajes."""
    return f"{x:.{decimales}f}".replace(".", ",").replace("-", "−")


def _sobrantes(sueltos, otros, nombres, ejemplo):
    """Avisa de valores sin nombre o con otro nombre. True si hubo alguno."""
    if sueltos:
        print(f"✗  Pasa los valores por nombre, como en la plantilla: {ejemplo}")
        return True
    if otros:
        print(f"✗  No conozco {', '.join(repr(k) for k in otros)}: usa estos nombres, en "
              f"minúsculas y sin tildes: {', '.join(n + '=' for n in nombres)}")
        return True
    return False


def _un_numero(x):
    """(valor, None) si x es un solo número real; si no, (None, motivo)."""
    import math
    import numbers
    import numpy as np

    if x is Ellipsis:
        return None, "todavía tiene los puntos suspensivos (...) de la plantilla"
    if x is None:
        return None, "no devuelve nada (None): ¿falta el return?"
    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.ravel()[0]
    if isinstance(x, (np.ndarray, list, tuple)):
        return None, (f"devuelve {len(x)} valores; hace falta uno solo, el valor nuevo de Q en esa casilla y acción. "
                      "El valor de la casilla s y la acción a se escribe Q[s][a]")
    if isinstance(x, (bool, np.bool_)):
        return None, "devuelve True o False; hace falta un número"
    if isinstance(x, str):
        return None, "devuelve texto; hace falta un número"
    if not isinstance(x, (numbers.Real, np.number)) or isinstance(x, np.complexfloating):
        return None, f"no devuelve un número real (devuelve {type(x).__name__})"
    valor = float(x)
    if math.isnan(valor) or math.isinf(valor):
        return None, "devuelve nan o infinito: revisa la cuenta"
    return valor, None


# ---------------------------------------------------------------------------------------------
# La rejilla (copia propia de los verificadores)
# ---------------------------------------------------------------------------------------------

def _paso(s, a):
    f = min(max(s[0] + _MOV[a][0], 0), _FILAS - 1)
    c = min(max(s[1] + _MOV[a][1], 0), _COLUMNAS - 1)
    s2 = (f, c)
    if s2 == _META:
        return s2, 10.0, True
    if s2 in _POZOS:
        return s2, -10.0, True
    return s2, 0.0, False


def _entrena(eps, episodios=500, alfa=0.1, gamma=0.9, semilla=2026):
    """El mismo entrenamiento de la celda [entrenar]."""
    import numpy as np

    rng = np.random.default_rng(semilla)
    Q = np.zeros((_FILAS, _COLUMNAS, 4))
    visitas = np.zeros((_FILAS, _COLUMNAS), dtype=int)
    for _ in range(episodios):
        s = _INICIO
        for _t in range(50):
            visitas[s] += 1
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                a = int(np.argmax(Q[s]))
            s2, r, fin = _paso(s, a)
            mejor_despues = 0 if fin else Q[s2].max()
            Q[s][a] += alfa * (r + gamma * mejor_despues - Q[s][a])
            s = s2
            if fin:
                break
    return Q, visitas


def _llega(Q):
    import numpy as np

    s = _INICIO
    for _ in range(50):
        s, _r, fin = _paso(s, int(np.argmax(Q[s])))
        if fin:
            return s == _META
    return False


# ---------------------------------------------------------------------------------------------
# Ejercicio 1 · La actualización de Q
# ---------------------------------------------------------------------------------------------

def _probar(funcion, Q, s, a, r, s2, fin, alfa, gamma):
    """Llama a la función del estudiante con una copia de Q. Devuelve (valor, motivo, cambió_Q)."""
    import numpy as np

    import inspect

    copia = Q.copy()
    try:
        inspect.signature(funcion).bind(copia, s, a, r, s2, fin, alfa=alfa, gamma=gamma)
    except TypeError:
        return None, ("no recibe los mismos argumentos de la plantilla: deja la primera línea como "
                      "estaba, def actualiza(Q, s, a, r, s2, fin, alfa=0.1, gamma=0.9):"), False
    except ValueError:
        pass   # una función sin firma legible: se prueba igual
    try:
        resultado = funcion(copia, s, a, r, s2, fin, alfa=alfa, gamma=gamma)
    except TypeError as error:
        texto = str(error)
        if "llipsis" in texto:
            return None, "todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos", False
        return None, f"dio un error al ejecutarla (TypeError: {texto})", False
    except SystemExit:
        return None, "detuvo la ejecución (SystemExit): quita el exit() o el quit()", False
    except NameError as error:
        return None, (f"usa un nombre que no existe dentro de la función ({error}): usa solo Q, s, a, r, s2, "
                      "fin, alfa, gamma y lo_mejor_despues"), False
    except Exception as error:   # noqa: BLE001
        return None, f"dio un error al ejecutarla ({type(error).__name__}: {error})", False
    cambio = not np.array_equal(copia, Q)
    valor, motivo = _un_numero(resultado)
    return valor, motivo, cambio


@_seguro
def revisar_actualiza(actualiza=...):
    """Ejercicio 1: la función que devuelve el valor nuevo de Q[s][a]."""
    import numpy as np

    if actualiza is Ellipsis or not callable(actualiza):
        print("✗  Pasa la función, sin paréntesis: revisar_actualiza(actualiza)")
        return
    # Caso 1: Q[s][a] = 2,0; r = −1; lo mejor del estado siguiente = 5,0; α = 0,1; γ = 0,9
    Q = np.zeros((_FILAS, _COLUMNAS, 4))
    s, a, s2 = (2, 0), 2, (1, 0)
    Q[s] = [3.0, 0.0, 2.0, -1.0]      # Q[s][a] = 2,0, que no es el máximo de su casilla
    Q[s2] = [1.0, 0.0, -3.0, 5.0]     # el máximo de la casilla siguiente no está en la acción a
    v, motivo, cambio = _probar(actualiza, Q, s, a, -1.0, s2, False, 0.1, 0.9)
    if motivo:
        print(f"✗  Tu función {motivo}")
        return
    if cambio:
        print("·  Tu función cambia la tabla Q que recibe. No hace falta: basta con que devuelva el valor nuevo")
    errores = {
        2.15: None,
        2.25: "el objetivo no incluye la recompensa que el agente acaba de recibir (r)",
        2.20: "lo que espera ganar después entra completo al objetivo: falta descontarlo con gamma",
        1.70: ("al objetivo le falta lo que espera ganar después: lo mejor del estado siguiente "
               "(lo_mejor_despues)"),
        3.50: "el valor salta de una vez hasta el objetivo: revisa el papel de alfa en la regla",
        2.35: ("revisa qué multiplica alfa: no es el objetivo entero, sino cuánto le falta al valor actual para "
               "llegar a él (con sus signos)"),
        0.15: "devuelve solo el cambio; falta sumarlo al valor actual",
        1.85: "resta en el orden contrario: lo que le falta al valor actual es el objetivo menos el valor actual",
    }
    pista = next((p for valor, p in errores.items() if abs(v - valor) < 1e-6), "sin pista")
    if pista is None:
        print("✓  Con un caso a mitad del camino: bien")
    else:
        print(f"✗  Con un caso a mitad del camino, tu función da {_coma(v)}. "
              + (pista[0].upper() + pista[1:] if pista != "sin pista" else
                 "Repasa la regla: el valor actual más alfa por (el objetivo menos el valor actual), y el "
                 "objetivo es la recompensa más gamma por lo mejor del estado siguiente"))
        return
    # Caso 2: los mismos datos con α = 0,5 y γ = 0,5 (¿usa los parámetros o números escritos a mano?)
    v2, motivo, _ = _probar(actualiza, Q, s, a, -1.0, s2, False, 0.5, 0.5)
    if motivo or abs(v2 - 1.75) > 1e-6:
        print("✗  Con otra alfa y otra gamma, tu función da lo mismo o no funciona: usa los parámetros alfa y "
              "gamma, no 0.1 ni 0.9 escritos a mano")
        return
    print("✓  Con otra tasa de aprendizaje y otro factor de descuento: bien")
    # Caso 3: el paso que termina (fin=True), con una fila no nula en el estado siguiente
    Qf = np.zeros((_FILAS, _COLUMNAS, 4))
    sf, af = (0, 2), 2
    Qf[sf] = [3.0, 0.0, 1.0, 0.0]
    Qf[_META] = [4.0, 4.0, 4.0, 4.0]
    v3, motivo, _ = _probar(actualiza, Qf, sf, af, 10.0, _META, True, 0.1, 0.9)
    if motivo:
        print(f"✗  En el último paso de un episodio, tu función {motivo}")
        return
    if abs(v3 - 1.9) > 1e-6:
        print("✗  En el último paso de un episodio (fin es True) no da lo esperado: al llegar a la meta o a un "
              "pozo ya no hay después, y lo que espera ganar después vale 0")
        return
    print("✓  En el último paso de un episodio: bien")
    print("\n¡Listo! Tu función hace lo mismo que la línea de la actualización en la celda [entrenar]: el valor "
          "se mueve una fracción alfa hacia el objetivo, r + gamma · (lo mejor de después).")


# ---------------------------------------------------------------------------------------------
# Ejercicio 2 · Sin explorar
# ---------------------------------------------------------------------------------------------

def _un_logico(x, nombre):
    import numpy as np

    if x is Ellipsis:
        return None, f"{nombre}: todavía tiene los puntos suspensivos (...) de la plantilla"
    if isinstance(x, (bool, np.bool_)):
        return bool(x), None
    if isinstance(x, str):
        return None, f"{nombre}: escribe True o False sin comillas (es un valor lógico, no un texto)"
    if isinstance(x, (int, np.integer)) and x in (0, 1):
        return None, f"{nombre}: escribe True o False, no 0 ni 1"
    return None, f"{nombre}: hace falta True o False, lo que devuelve llega_a_la_meta(Q)"


@_seguro
def revisar_exploracion(*sueltos, llega_eps0=..., llega_eps01=..., casillas_eps0=..., razon=..., **otros):
    """Ejercicio 2: ¿llega cada agente a la meta? ¿Por qué el que no explora no llega?"""
    nombres = ["llega_eps0", "llega_eps01", "casillas_eps0", "razon"]
    if _sobrantes(sueltos, otros, nombres,
                  "revisar_exploracion(llega_eps0=llega_eps0, llega_eps01=llega_eps01, "
                  "casillas_eps0=casillas_eps0, razon=razon)"):
        return
    if "rejilla" not in _CACHE:
        q0, v0 = _entrena(0.0)
        q01, _v01 = _entrena(0.1)
        _CACHE["rejilla"] = {"0": _llega(q0), "01": _llega(q01), "visitadas0": int((v0 > 0).sum()),
                             "pasos0": int(v0.sum())}
    ref = _CACHE["rejilla"]
    bien = 0
    for nombre, x, clave in (("llega_eps0", llega_eps0, "0"), ("llega_eps01", llega_eps01, "01")):
        valor, motivo = _un_logico(x, nombre)
        if motivo:
            print(f"✗  {motivo}")
        elif valor == ref[clave]:
            print(f"✓  {nombre}: bien")
            bien += 1
        else:
            print(f"✗  {nombre}: no es lo que pasa con ese agente. ¿Lo entrenaste con el ε que dice su nombre y la "
                  "semilla de la plantilla? ¿Le pasaste a llega_a_la_meta la tabla Q de ese mismo agente?")
    import numbers
    import numpy as np

    if casillas_eps0 is Ellipsis:
        print("✗  casillas_eps0: todavía tiene los puntos suspensivos (...)")
    elif isinstance(casillas_eps0, np.ndarray) and casillas_eps0.size > 1:
        print("✗  casillas_eps0: es una tabla; hace falta un solo número: súmala con .sum()")
    elif isinstance(casillas_eps0, (bool, np.bool_)) or not isinstance(casillas_eps0, (numbers.Integral, np.integer)):
        print("✗  casillas_eps0: hace falta un número entero, cuántas casillas tienen al menos una visita")
    elif int(casillas_eps0) == ref["visitadas0"]:
        print("✓  casillas_eps0: bien")
        bien += 1
    elif int(casillas_eps0) == ref["pasos0"]:
        print("✗  casillas_eps0: ese es el total de pasos; se piden las casillas distintas con al menos una visita")
    else:
        print("✗  casillas_eps0: no es lo que da el agente sin exploración: cuenta las casillas de visitas_0 que "
              "no están en 0")
    pistas = {
        "A": ("La tasa de aprendizaje es la misma en los dos agentes: lo único que cambia es ε. Mira las visitas "
              "del agente que no explora: ¿por cuáles casillas pasó?"),
        "B": ("Con γ = 0,9, llegar a la meta en 6 pasos todavía vale 10 · 0,9⁵ ≈ 5,9, más que cero: sí vale la "
              "pena ir. Mira las visitas del agente que no explora: ¿llegó alguna vez cerca de la meta?"),
        "D": ("Pruébalo: entrena el agente con eps=0.0 y episodios=5000. ¿Llega ahora? Mira también cuántas "
              "casillas distintas visitó"),
    }
    if razon is Ellipsis:
        print("✗  razon: todavía tiene los puntos suspensivos (...): escribe la letra entre comillas, por ejemplo \"A\"")
    elif not isinstance(razon, str) or razon.strip().upper().strip(".)") not in ("A", "B", "C", "D"):
        print("✗  razon: escribe solo la letra de la explicación, entre comillas: \"A\", \"B\", \"C\" o \"D\"")
    else:
        letra = razon.strip().upper().strip(".)")
        if letra == _RAZON_BIEN:
            print("✓  razon: bien")
            bien += 1
        else:
            print(f"✗  razon: no es esa. {pistas[letra]}")
    if bien == 4:
        print(f"\n¡Listo! Sin explorar, el agente visitó solo {ref['visitadas0']} casillas en 500 episodios: como "
              "nunca recibió una recompensa, su tabla quedó en ceros y eligió siempre la misma acción. Con "
              "ε = 0,1, las jugadas al azar lo llevan alguna vez a la meta, y desde ahí la recompensa viaja hacia atrás.")


# ---------------------------------------------------------------------------------------------
# Ejercicio 3 · Tu política de retención (simulador propio, con los mismos datos y supuestos)
# ---------------------------------------------------------------------------------------------

def _yarumo():
    """El riesgo mensual por tramo, la factura y los supuestos, como en las celdas [riesgo] y [entorno]."""
    if "yarumo" in _CACHE:
        return _CACHE["yarumo"]
    import pandas as pd

    try:
        clientes = pd.read_csv("datos/clientes.csv")
    except FileNotFoundError:
        print("✗  No encuentro datos/clientes.csv: ejecuta primero la celda que descarga los datos "
              "(Entorno de ejecución → Ejecutar anteriores)")
        return None
    ant = clientes["antiguedad_meses"]
    sel = clientes[((ant <= 12) & (clientes["descuento"] == "Sí"))
                   | (ant.between(13, 24) & (clientes["descuento"] == "No"))]
    tramo = pd.cut(sel["antiguedad_meses"], [0, 9, 12, 24], labels=False)
    a3 = (sel["cancelo"] == "Sí").groupby(tramo).mean().to_numpy()
    _CACHE["yarumo"] = {"riesgo": 1 - (1 - a3) ** (1 / 3),
                        "factura": round(clientes["factura_mensual"].median() / 1000),
                        "baja": (0.0, 0.25, 0.60), "costo": (0.0, 20.0, 60.0)}
    return _CACHE["yarumo"]


def _valor_exacto(acciones):
    """Valor esperado por cliente (miles de pesos) de una lista de 24 acciones, de atrás hacia adelante."""
    y = _yarumo()
    valor = 0.0
    for mes in range(24, 0, -1):
        a = acciones[mes - 1]
        k = 0 if mes <= 9 else (1 if mes <= 12 else 2)
        riesgo = y["riesgo"][k] * (1 - y["baja"][a])
        valor = -y["costo"][a] + (1 - riesgo) * (y["factura"] + valor)
    return valor


def _acciones_de(politica):
    """Las 24 acciones de la política (función del mes o lista), o (None, motivo)."""
    import numbers
    import numpy as np

    if politica is Ellipsis:
        return None, "tu política todavía tiene los puntos suspensivos (...)"
    if callable(politica):
        acciones = []
        for mes in range(1, 25):
            try:
                a = politica(mes)
            except TypeError as error:
                if "llipsis" in str(error):
                    return None, "tu política todavía tiene los puntos suspensivos (...): reemplázalos"
                return None, f"tu política dio un error en el mes {mes} (TypeError: {error})"
            except Exception as error:   # noqa: BLE001
                return None, f"tu política dio un error en el mes {mes} ({type(error).__name__}: {error})"
            acciones.append(a)
    elif isinstance(politica, (list, tuple, np.ndarray)):
        acciones = list(np.ravel(politica)) if isinstance(politica, np.ndarray) else list(politica)
        if len(acciones) != 24:
            return None, f"tu lista tiene {len(acciones)} acciones; hacen falta 24, una por mes"
    else:
        return None, "pasa tu política, la función, sin paréntesis: revisar_politica(mi_politica, valor=valor)"
    for mes, a in enumerate(acciones, start=1):
        if a is Ellipsis:
            return None, f"en el mes {mes} tu política devuelve los puntos suspensivos (...) de la plantilla"
        if a is None:
            return None, f"en el mes {mes} tu política no devuelve nada: ¿falta un return?"
        if isinstance(a, str):
            return None, (f"en el mes {mes} tu política devuelve el texto {a!r}: devuelve el número de la acción, "
                          "0 (nada), 1 (llamar) o 2 (oferta de retención)")
        if isinstance(a, (bool, np.bool_)) or not isinstance(a, (numbers.Integral, np.integer)):
            return None, (f"en el mes {mes} tu política devuelve {a!r}: devuelve 0, 1 o 2, números enteros")
        if int(a) not in (0, 1, 2):
            return None, f"en el mes {mes} tu política devuelve {a}: las acciones son 0, 1 y 2"
    return [int(a) for a in acciones], None


@_seguro
def revisar_politica(politica=..., *sueltos, valor=..., **otros):
    """Ejercicio 3: una política de retención válida y su valor, calculado con valor_politica."""
    import numpy as np

    if _sobrantes(sueltos, otros, ["valor"], "revisar_politica(mi_politica, valor=valor)"):
        return
    if _yarumo() is None:
        return
    acciones, motivo = _acciones_de(politica)
    if motivo:
        print(f"✗  {motivo}")
        return
    print("✓  Tu política da una acción válida en los 24 meses")
    nombres = {0: "nada", 1: "llamar", 2: "oferta de retención"}
    tramos = []
    inicio = 1
    for mes in range(2, 26):
        if mes == 25 or acciones[mes - 1] != acciones[inicio - 1]:
            fin = mes - 1
            tramos.append(f"{nombres[acciones[inicio - 1]]} en {'el mes ' + str(inicio) if inicio == fin else f'los meses {inicio}–{fin}'}")
            inicio = mes
    print("   Tu plan: " + "; ".join(tramos))
    exacto = _valor_exacto(acciones)
    v, problema = _un_numero(valor)
    if problema:
        if valor is Ellipsis:
            problema = "todavía tiene los puntos suspensivos (...)"
        elif valor is None:
            problema = "está vacío (None)"
        elif isinstance(valor, (list, tuple, np.ndarray)):
            problema = "es una lista o una tabla; hace falta un solo número"
        elif isinstance(valor, str):
            problema = "es texto; hace falta un número"
        else:
            problema = "no es un número"
        print(f"✗  valor: {problema}. Calcúlalo con valor_politica(mi_politica)")
        return
    if abs(v - exacto) <= 0.05:
        print("✓  valor: bien")
    elif abs(v - round(exacto)) < 1e-9 or abs(v - round(exacto, 1)) < 1e-9:
        print("✓  valor: bien (redondeado)")
    else:
        otros_valores = {"no hacer nada": [0] * 24, "llamar siempre": [1] * 24, "la oferta siempre": [2] * 24,
                         "la oferta en los meses 10–12": [2 if 10 <= m <= 12 else 0 for m in range(1, 25)]}
        de_otra = next((n for n, acc in otros_valores.items() if acc != acciones and abs(_valor_exacto(acc) - v) < 0.06),
                       None)
        if de_otra:
            print(f"✗  valor: es el de otra política ({de_otra}); calcula el de la tuya con valor_politica(mi_politica)")
        else:
            print("✗  valor: no coincide con el de tu política. No lo escribas a mano: calcúlalo con "
                  "valor_politica(mi_politica)")
        return
    regla = _valor_exacto([2 if 10 <= m <= 12 else 0 for m in range(1, 25)])
    print(f"\n¡Listo! Bajo los supuestos del simulador, tu política deja en promedio {_coma(exacto, 1)} mil pesos por "
          f"cliente en 24 meses; la oferta solo en los meses 10–12 deja {_coma(regla, 1)} mil pesos. Compárala con la mejor "
          "posible, en la celda [mejor], y explica en «Tus respuestas» por qué la tuya gana o pierde.")


# ---------------------------------------------------------------------------------------------
# Dibujos (los llama el notebook; no hace falta leerlos)
# ---------------------------------------------------------------------------------------------

def dibujar_rejilla(Q=None, visitas=None, titulo=""):
    """La rejilla: la salida, la meta, los pozos y, si se pasa Q, la mejor acción de cada casilla
    (una flecha) con las veces que el agente pasó por ella debajo. Un punto gris: empate, nada aprendido."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    fig, eje = plt.subplots(figsize=(4.6, 4.6) if Q is None else (5.2, 5.2))
    for f in range(_FILAS):
        for c in range(_COLUMNAS):
            s = (f, c)
            if s == _META:
                fondo, texto, color = "#FDE5D3", "Meta\n+10", "#B44700"
            elif s in _POZOS:
                fondo, texto, color = "#566B7F", "Pozo\n−10", "#FFFFFF"
            elif s == _INICIO:
                fondo, texto, color = "#EFE3F3", "Salida", "#6B2179"
            else:
                fondo, texto, color = "#FFFFFF", "", "#1E293B"
            eje.add_patch(Rectangle((c, f), 1, 1, facecolor=fondo, edgecolor="#94A3B8", linewidth=1.5))
            libre = s != _META and s not in _POZOS
            if Q is None or not libre:
                eje.text(c + 0.5, f + 0.5, texto, ha="center", va="center", color=color,
                         fontsize=14, fontweight="bold")
                continue
            if s == _INICIO:
                eje.text(c + 0.5, f + 0.14, "Salida", ha="center", va="center", color=color, fontsize=10,
                         fontweight="bold")
            if np.allclose(Q[s], Q[s][0]):
                eje.plot(c + 0.5, f + 0.45, "o", color="#94A3B8", markersize=7)
            else:
                df, dc = _MOV[int(np.argmax(Q[s]))]
                eje.annotate("", xy=(c + 0.5 + 0.27 * dc, f + 0.45 + 0.22 * df),
                             xytext=(c + 0.5 - 0.27 * dc, f + 0.45 - 0.22 * df),
                             arrowprops=dict(arrowstyle="-|>", color="#8A2E9E", lw=3, mutation_scale=22))
            if visitas is not None:
                eje.text(c + 0.5, f + 0.84, f"{int(visitas[s])}", ha="center", va="center", color="#475569",
                         fontsize=11)
    eje.set_xlim(0, _COLUMNAS)
    eje.set_ylim(_FILAS, 0)
    eje.set_aspect("equal")
    eje.set_xticks([x + 0.5 for x in range(_COLUMNAS)], [str(x) for x in range(_COLUMNAS)])
    eje.set_yticks([y + 0.5 for y in range(_FILAS)], [str(y) for y in range(_FILAS)])
    eje.tick_params(length=0)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_visible(False)
    eje.set_xlabel("Columna")
    eje.set_ylabel("Fila")
    if titulo:
        eje.set_title(titulo, fontsize=13)
    plt.show()


def dibujar_planes(planes, titulo=""):
    """Una fila por política: el color de cada mes dice qué acción toma (nada, llamar u oferta)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    colores = {0: "#E2E8F0", 1: "#8A2E9E", 2: "#F46201"}
    nombres = {0: "Nada", 1: "Llamar", 2: "Oferta de retención"}
    fig, eje = plt.subplots(figsize=(10.5, 0.75 * len(planes) + 1.5))
    for i, (nombre, acciones) in enumerate(planes.items()):
        for mes, a in enumerate(acciones, start=1):
            eje.add_patch(Rectangle((mes - 0.5, i - 0.4), 1, 0.8, facecolor=colores[a], edgecolor="#FFFFFF",
                                    linewidth=1.5))
    eje.set_xlim(0.5, len(next(iter(planes.values()))) + 0.5)
    eje.set_ylim(len(planes) - 0.5, -0.5)
    eje.set_yticks(range(len(planes)), list(planes))
    eje.set_xticks([1, 5, 10, 12, 18, 24])
    eje.set_xlabel("Mes del cliente")
    eje.tick_params(length=0)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_visible(False)
    eje.axvspan(9.5, 12.5, ymin=0, ymax=1, facecolor="none", edgecolor="#1E293B", linewidth=1.2, linestyle="--")
    eje.legend(handles=[Patch(facecolor=colores[a], label=nombres[a]) for a in (0, 1, 2)], loc="upper center",
               bbox_to_anchor=(0.5, -0.32 if len(planes) > 2 else -0.5), ncol=3, frameon=False)
    if titulo:
        eje.set_title(titulo, fontsize=13)
    plt.show()
