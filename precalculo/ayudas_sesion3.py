"""Sesión 3: estilo de las gráficas del curso y los verificadores de la práctica.

Van dentro de la celda oculta del notebook, junto con descarga_datos.py y ayudas.py. Cada
verificador recibe lo que el estudiante calculó, dice qué está bien y, en lo demás, da una
pista sin la respuesta. Nunca se cae: un «...» sin reemplazar o un tipo inesperado se avisa.

Los valores esperados se calculan aquí, en tiempo de ejecución, desde datos/clientes.csv, con
las mismas columnas, el mismo escalado y la misma semilla de la clase (y se guardan para no
repetir el cálculo). Los grupos se reconocen por su perfil (edad, usuarios, pymes), nunca por el
número de etiqueta, que puede cambiar con otra versión de scikit-learn.
Tolerancias: ejercicio 1, ±0,005 en cada silueta; ejercicio 2, ±1,5 puntos en la tasa;
ejercicio 3, conteos exactos y ±0,5 puntos en la proporción de pymes; ejercicio 4, solo pistas.
"""


def _estilo_graficas():
    """Colores del curso y letras que se leen en Colab y proyectadas."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=["#8A2E9E", "#F46201", "#566B7F"]),
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

_CACHE = {}   # lo que ya se calculó: los datos escalados y los agrupamientos de referencia
_COLUMNAS = ["edad", "usuarios", "consumo_gb"]


def _datos():
    """Los clientes, sus tres columnas sin escalar y escaladas; o None con aviso."""
    if "datos" in _CACHE:
        return _CACHE["datos"]
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    try:
        clientes = pd.read_csv("datos/clientes.csv")
    except FileNotFoundError:
        print("✗  No encuentro datos/clientes.csv: ejecuta primero la celda que descarga los datos "
              "(Entorno de ejecución → Ejecutar anteriores)")
        return None
    X = clientes[_COLUMNAS]
    _CACHE["datos"] = {"clientes": clientes, "X": X, "Z": StandardScaler().fit_transform(X)}
    return _CACHE["datos"]


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


def _seguro(verificador):
    """Red de seguridad: si algo imprevisto falla al revisar, se avisa sin traceback."""
    import functools

    @functools.wraps(verificador)
    def envoltura(*args, **kwargs):
        try:
            return verificador(*args, **kwargs)
        except Exception as error:   # noqa: BLE001 — el estudiante nunca ve un traceback
            print(f"✗  No pude revisar lo que entregaste ({type(error).__name__}). Revisa que cada valor "
                  "sea el resultado de un cálculo y vuelve a ejecutar la celda")
    return envoltura


def _un_numero(x):
    """(valor, None) si x es un número (o algo con un solo número); si no, (None, motivo)."""
    import math
    import numbers
    import numpy as np
    import pandas as pd

    if x is Ellipsis:
        return None, "todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos por tu cálculo"
    if isinstance(x, pd.DataFrame):
        if x.size != 1:
            forma = f"{x.shape[0]} × {x.shape[1]}" if x.shape[0] <= 10 else "muchas filas"   # no regalar un conteo
            return None, f"es una tabla de {forma}; hace falta un solo número (¿te faltó contar o promediar?)"
        x = x.iloc[0, 0]
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple, dict)):
        valores = list(x.values()) if isinstance(x, dict) else list(np.ravel(np.asarray(x, dtype=object)))
        if len(valores) == 0:
            return None, "está vacío"
        if len(valores) > 1:
            if len(valores) == 8000:
                return None, "es un valor por cliente; falta resumirlo en un número"
            cuantos = f"{len(valores)} valores" if len(valores) <= 10 else "muchos valores"   # no regalar un conteo
            return None, f"tiene {cuantos}; hace falta uno solo (¿te faltó contar con len() o sum(), o promediar con mean()?)"
        x = valores[0]
        if x is Ellipsis:
            return None, "todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos por tu cálculo"
    if x is None:
        return None, "está vacío (None): ¿le asignaste el resultado de un print?"
    if callable(x):
        return None, "es una función, no su resultado: ¿faltaron los paréntesis o los datos?"
    if isinstance(x, (bool, np.bool_)):
        return None, "es True o False; hace falta un número"
    if isinstance(x, str):
        return None, "es texto; escríbelo como número, sin comillas y con punto decimal"
    if hasattr(x, "fit") and hasattr(x, "get_params"):
        return None, "es un modelo, no un número: ¿te faltó calcular algo con él?"
    if not isinstance(x, (numbers.Real, np.number)) or isinstance(x, np.complexfloating):
        return None, f"no es un número real (es {type(x).__name__})"
    try:
        valor = float(x)
    except OverflowError:
        return None, "es un número demasiado grande"
    if math.isnan(valor):
        return None, "salió nan: revisa el cálculo"
    if math.isinf(valor):
        return None, "salió infinito: ¿dividiste entre cero?"
    return valor, None


def _leer(entregados, nombres):
    """Convierte cada valor entregado en número; imprime los problemas. Devuelve {nombre: valor}."""
    valores, problemas = {}, {}
    for nombre in nombres:
        x = entregados[nombre]
        if x is Ellipsis:
            print(f"·  {nombre}: todavía tiene los puntos suspensivos (...) de la plantilla: reemplázalos por tu cálculo")
            continue
        valor, problema = _un_numero(x)
        if problema:
            problemas.setdefault(problema, []).append(nombre)
        else:
            valores[nombre] = valor
    for problema, quienes in problemas.items():   # la misma pista una sola vez
        print(f"✗  {', '.join(quienes)}: {problema}")
    return valores


def _etiquetas(x, nombre):
    """Un arreglo de enteros con una etiqueta por cliente, o None con aviso."""
    import numpy as np
    import pandas as pd

    if x is Ellipsis:
        print(f"·  {nombre}: todavía tiene los puntos suspensivos (...) de la plantilla")
        return None
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        x = x.iloc[:, 0]
    try:
        arreglo = np.asarray(x)
    except Exception:   # noqa: BLE001
        arreglo = None
    if arreglo is None or arreglo.ndim != 1 or len(arreglo) != 8000:
        print(f"✗  {nombre}: tiene que ser el número de grupo de cada uno de los 8000 clientes "
              "(lo que devuelve fit_predict)")
        return None
    if not np.issubdtype(arreglo.dtype, np.integer):
        try:
            if np.all(np.mod(arreglo.astype(float), 1) == 0):
                arreglo = arreglo.astype(int)
            else:
                raise ValueError
        except (TypeError, ValueError):
            print(f"✗  {nombre}: tiene que traer números de grupo enteros (0, 1, 2…), como los que devuelve "
                  "fit_predict")
            return None
    return arreglo


def _grupos(n):
    """«1 grupo», «4 grupos»."""
    return f"{int(n)} grupo" + ("" if int(n) == 1 else "s")


def _coma(x, decimales=3):
    """Un número con coma decimal, para los mensajes."""
    return f"{x:.{decimales}f}".replace(".", ",").replace("-", "−")


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


# ---------------------------------------------------------------------------------------------
# Los agrupamientos de referencia
# ---------------------------------------------------------------------------------------------

def _kmeans(datos, k, n_init=10):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=n_init, random_state=2026).fit_predict(datos)


def _ref_siluetas(variante="bien"):
    """Siluetas de k = 2…8: «bien» (Z, n_init=10: la clase), «sin_escalar» (X) o «auto» (una sola corrida).
    Cada variante se calcula la primera vez que se pide: la respuesta correcta solo paga la primera."""
    clave = ("siluetas", variante)
    if clave not in _CACHE:
        from sklearn.metrics import silhouette_score

        d = _datos()
        datos = d["X"] if variante == "sin_escalar" else d["Z"]
        n_init = "auto" if variante == "auto" else 10
        _CACHE[clave] = {k: silhouette_score(datos, _kmeans(datos, k, n_init)) for k in range(2, 9)}
    return _CACHE[clave]


def _ref_grupos():
    """Los 4 grupos de la clase (escalados) y los 4 sin escalar."""
    if "grupos" in _CACHE:
        return _CACHE["grupos"]
    d = _datos()
    _CACHE["grupos"] = {"bien": _kmeans(d["Z"], 4), "sin_escalar": _kmeans(d["X"], 4)}
    return _CACHE["grupos"]


def _perfil(grupo_mask):
    """Una descripción del grupo por sus promedios, sin decir su número."""
    d = _datos()
    X = d["X"][grupo_mask]
    pymes = (d["clientes"]["tipo_cliente"][grupo_mask] == "Pyme").mean()
    return (f"edad media de {_coma(X['edad'].mean(), 0)} años, {_coma(X['usuarios'].mean(), 1)} usuarios "
            f"y {_coma(pymes * 100, 0)} % de pymes")


# ---------------------------------------------------------------------------------------------
# Ejercicio 1 · ¿Cuántos grupos?
# ---------------------------------------------------------------------------------------------

def _siluetas_entregadas(siluetas):
    """{k: valor} desde un diccionario, una serie o una lista de 7 (k = 2…8); o None con aviso."""
    import numpy as np
    import pandas as pd

    if siluetas is Ellipsis:
        print("·  siluetas: todavía tiene los puntos suspensivos (...) de la plantilla")
        return None
    if isinstance(siluetas, pd.Series):
        siluetas = siluetas.to_dict()
    if isinstance(siluetas, (list, tuple, np.ndarray)):
        valores = list(np.ravel(np.asarray(siluetas, dtype=object)))
        if len(valores) != 7:
            print(f"✗  siluetas: tiene {len(valores)} valor{'' if len(valores) == 1 else 'es'}; van 7, uno por cada k de 2 a 8 (mejor, un "
                  "diccionario {k: silueta}, como en la plantilla)")
            return None
        siluetas = dict(zip(range(2, 9), valores))
    if not isinstance(siluetas, dict):
        valor, _ = _un_numero(siluetas)
        if valor is not None:
            print("✗  siluetas: es un solo número; van las 7 siluetas, una por cada k de 2 a 8, en el "
                  "diccionario siluetas")
        else:
            print("✗  siluetas: tiene que ser el diccionario {k: silueta} de la plantilla")
        return None
    salida, faltan = {}, []
    for k in range(2, 9):
        clave = k if k in siluetas else (str(k) if str(k) in siluetas else None)
        if clave is None:
            faltan.append(k)
            continue
        valor, problema = _un_numero(siluetas[clave])
        if problema:
            print(f"✗  siluetas[{k}]: {problema}")
            return None
        salida[k] = valor
    if faltan:
        print(f"✗  siluetas: faltan los k {', '.join(map(str, faltan))}; el ciclo va de 2 a 8 "
              "(range(2, 9) termina en 8)")
        return None
    return salida


@_seguro
def revisar_k(*sueltos, siluetas=..., k_silueta=..., k_gerente=..., **otros):
    """Ejercicio 1: las siluetas de k = 2…8, el k de la silueta más alta y el k para la gerente."""
    nombres = ["siluetas", "k_silueta", "k_gerente"]
    if _sobrantes(sueltos, otros, nombres, "revisar_k(siluetas=siluetas, k_silueta=k_silueta, "
                                           "k_gerente=k_gerente)"):
        return
    if _datos() is None:
        return
    ref = {"bien": _ref_siluetas("bien")}
    bien_todo = True
    s = _siluetas_entregadas(siluetas)
    if s is None:
        bien_todo = False
    else:
        malos = [k for k in s if abs(s[k] - ref["bien"][k]) > 0.005]
        if not malos:
            print("✓  siluetas: bien")
        else:
            bien_todo = False
            if all(abs(s[k] - _ref_siluetas("sin_escalar")[k]) <= 0.005 for k in s):
                print("✗  siluetas: son las de los datos sin escalar (X). Agrupa y mide con los datos "
                      "escalados, Z, en fit_predict y en silhouette_score")
            elif all(abs(s[k] - _ref_siluetas("auto")[k]) <= 0.005 for k in s):
                print("✗  siluetas: son las de una sola corrida de K-Means: pon n_init=10")
            elif all(1 < s[k] <= 100 for k in s):
                print("✗  siluetas: la silueta va de −1 a 1; no la escribas como porcentaje")
            elif all(abs(s[k] - ref["bien"][k]) <= 0.03 for k in s):
                print(f"✗  siluetas: casi; {', '.join(f'k = {k}' for k in malos)} no coinciden. Revisa que "
                      "uses n_init=10 y random_state=2026, como en la plantilla")
            elif any(s[k] > 1 or s[k] < -1 for k in s):
                print("✗  siluetas: una silueta siempre está entre −1 y 1: ¿guardaste la inercia u otra "
                      "cosa? Usa silhouette_score(Z, etiquetas_k)")
            else:
                print("✗  siluetas: no coinciden. Para cada k: entrena con Z, y mide "
                      "silhouette_score(Z, etiquetas_k) con las etiquetas de ese mismo k")
    mejor = max(ref["bien"], key=ref["bien"].get)
    vals = _leer({"k_silueta": k_silueta, "k_gerente": k_gerente}, ["k_silueta", "k_gerente"])
    ks = vals.get("k_silueta")
    if "k_silueta" not in vals:
        bien_todo = False
    elif ks != int(ks) or not 2 <= ks <= 8:
        print("✗  k_silueta: es uno de los k que probaste, un entero de 2 a 8")
        bien_todo = False
    elif int(ks) == mejor:
        print("✓  k_silueta: bien")
    else:
        bien_todo = False
        if int(ks) == 8:
            print("✗  k_silueta: ese es el k más grande que probaste, no el de la silueta más alta. Si usaste "
                  "max(siluetas): sobre un diccionario, max compara las claves; pídele que compare los valores "
                  "con key=siluetas.get")
        elif s is not None and int(ks) == max(s, key=s.get):
            print("✗  k_silueta: es el k de tu silueta más alta, pero tus siluetas no coinciden: corrígelas "
                  "primero")
        else:
            print(f"✗  k_silueta: con k = {int(ks)}, la silueta es {_coma(ref['bien'][int(ks)])}; hay un k con "
                  "una silueta más alta. La pregunta es solo por la silueta, no por lo que conviene")
    kg = vals.get("k_gerente")
    if "k_gerente" not in vals:
        bien_todo = False
    elif kg != int(kg) or kg < 1:
        print("✗  k_gerente: es un número de grupos: un entero positivo")
        bien_todo = False
    elif 3 <= kg <= 5:
        print(f"✓  k_gerente: {int(kg)} grupos caben en lo que la gerente puede atender")
    else:
        bien_todo = False
        if kg < 3:
            print(f"✗  k_gerente: con {_grupos(kg)} le quedaría por debajo de lo que la gerente puede atender "
                  "(de 3 a 5 ofertas distintas): ¿qué se pierde con un corte tan grueso?")
        else:
            print(f"✗  k_gerente: con {_grupos(kg)} se pasa de lo que la gerente puede atender (de 3 a 5 "
                  "ofertas distintas): ¿cuántas ofertas tendría que preparar?")
    if bien_todo:
        print(f"\n¡Listo! La silueta más alta es la de k = {mejor} ({_coma(ref['bien'][mejor])}). Escribe en "
              "«Tus respuestas» por qué recomiendas otro k a la gerente: ¿qué separa el corte de "
              f"{mejor} grupos? (Míralo con perfiles, como en la celda [perfiles].)")


# ---------------------------------------------------------------------------------------------
# Ejercicio 2 · ¿Cancelan igual?
# ---------------------------------------------------------------------------------------------

@_seguro
def revisar_cancelacion(*sueltos, grupos=..., grupo=..., tasa=..., **otros):
    """Ejercicio 2: el grupo (de los 4 de la clase) que más cancela y su tasa de cancelación."""
    import numpy as np

    nombres = ["grupos", "grupo", "tasa"]
    if _sobrantes(sueltos, otros, nombres, "revisar_cancelacion(grupos=grupos, grupo=grupo_mas, "
                                           "tasa=tasa_mas)"):
        return
    d = _datos()
    if d is None:
        return
    g = _etiquetas(grupos, "grupos")
    if g is None:
        return
    distintos = np.unique(g[g >= 0])
    if len(distintos) != 4:
        print(f"✗  grupos: tiene {_grupos(len(distintos))} y el ejercicio es con los 4 de la clase: vuelve a "
              "ejecutar la celda [kmeans4] (la variable grupos cambió)")
        return
    ref = _ref_grupos()
    if _ari(g, ref["bien"]) < 0.95:
        if _ari(g, ref["sin_escalar"]) >= 0.95:
            print("✗  grupos: esos son los grupos sin escalar; usa los de la celda [kmeans4], hechos con Z")
        else:
            print("✗  grupos: no son los 4 grupos de la clase: vuelve a ejecutar la celda [kmeans4]")
        return
    cancela = (d["clientes"]["cancelo"] == "Sí").to_numpy()
    tasas = {int(e): cancela[g == e].mean() for e in distintos}
    el_mas = max(tasas, key=tasas.get)
    vals = _leer({"grupo": grupo, "tasa": tasa}, ["grupo", "tasa"])
    bien_g = bien_t = False
    v = vals.get("grupo")
    if v is not None:
        if v != int(v) or int(v) not in tasas:
            print(f"✗  grupo: es el número de uno de los grupos ({', '.join(map(str, sorted(tasas)))}), no "
                  "su tasa ni su tamaño")
        elif int(v) == el_mas:
            bien_g = True
            print("✓  grupo: bien")
        else:
            print(f"✗  grupo: el grupo {int(v)} (el de {_perfil(g == int(v))}) cancela el "
                  f"{_coma(tasas[int(v)] * 100, 1)} %; hay uno que cancela más")
    t = vals.get("tasa")
    if t is not None:
        objetivo = tasas[el_mas]
        if abs(t - objetivo) <= 0.015:
            bien_t = True
            print("✓  tasa: bien")
        elif 1 < t <= 100 and abs(t / 100 - objetivo) <= 0.015:
            bien_t = True
            print("✓  tasa: bien, aunque la escribiste como porcentaje; como proporción va entre 0 y 1")
        elif abs(t - cancela.mean()) <= 0.002 or abs(t / 100 - cancela.mean()) <= 0.002:
            print("✗  tasa: esa es la tasa de todos los clientes; va la del grupo que más cancela "
                  "(agrupa con .groupby(grupos) antes de promediar)")
        elif t == int(t) and t > 100:
            print("✗  tasa: parece un conteo de clientes; la tasa es una proporción: el promedio de cancela "
                  "dentro del grupo")
        else:
            otro = [e for e, r in tasas.items() if e != el_mas and (abs(t - r) <= 0.002 or abs(t / 100 - r) <= 0.002)]
            if otro:
                print(f"✗  tasa: esa es la tasa del grupo {otro[0]}, no la del que más cancela")
            elif abs(t - (1 - objetivo)) <= 0.002:
                print("✗  tasa: esa es la proporción de los que NO cancelaron: compara con \"Sí\"")
            else:
                print("✗  tasa: no coincide: es el promedio de cancela (True o False) entre los clientes de ese "
                      "grupo")
    if bien_g and bien_t:
        resto = sorted(r for e, r in tasas.items() if e != el_mas)
        print(f"\n¡Listo! El grupo de {_perfil(g == el_mas)} cancela el {_coma(tasas[el_mas] * 100, 1)} %; los "
              f"otros, entre el {_coma(resto[0] * 100, 1)} y el {_coma(resto[-1] * 100, 1)} %. Recuerda que "
              "cancelo no entró al agrupamiento: los grupos no se buscaron para separar a los que cancelan, y aun "
              "así cancelan distinto. Escribe en «Tus respuestas» qué perfil es y qué le propondrías a la gerente.")


# ---------------------------------------------------------------------------------------------
# Ejercicio 3 · Lo raro (DBSCAN)
# ---------------------------------------------------------------------------------------------

def _ref_dbscan():
    """Grupos, ruido y pymes del ruido para eps 0,3 y 0,5: la clase y los errores típicos."""
    if "dbscan" in _CACHE:
        return _CACHE["dbscan"]
    import numpy as np
    from sklearn.cluster import DBSCAN

    d = _datos()
    pyme = (d["clientes"]["tipo_cliente"] == "Pyme").to_numpy()
    ref = {}
    for nombre, datos, ms in (("bien", d["Z"], 20), ("ms5", d["Z"], 5), ("sin_escalar", d["X"], 20)):
        ref[nombre] = {}
        for eps in (0.3, 0.5):
            e = DBSCAN(eps=eps, min_samples=ms).fit_predict(datos)
            r = e == -1
            ref[nombre][eps] = {"grupos": int(e.max() + 1), "ruido": int(r.sum()),
                                "pymes": float(pyme[r].mean()) if r.any() else float("nan"),
                                "pymes_ruido_de_pymes": float(r[pyme].mean())}
    _CACHE["dbscan"] = ref
    return ref


@_seguro
def revisar_ruido(*sueltos, grupos_03=..., ruido_03=..., grupos_05=..., ruido_05=..., pymes_ruido=...,
                  **otros):
    """Ejercicio 3: grupos y ruido de DBSCAN con eps 0,3 y 0,5 (min_samples=20), y las pymes del ruido."""
    nombres = ["grupos_03", "ruido_03", "grupos_05", "ruido_05", "pymes_ruido"]
    if _sobrantes(sueltos, otros, nombres, "revisar_ruido(grupos_03=grupos_03, ruido_03=ruido_03, "
                                           "grupos_05=grupos_05, ruido_05=ruido_05, pymes_ruido=pymes_ruido)"):
        return
    if _datos() is None:
        return
    vals = _leer({"grupos_03": grupos_03, "ruido_03": ruido_03, "grupos_05": grupos_05,
                  "ruido_05": ruido_05, "pymes_ruido": pymes_ruido}, nombres)
    if not vals:
        return
    ref = _ref_dbscan()
    bien = {}
    for eps, sufijo in ((0.3, "03"), (0.5, "05")):
        b = ref["bien"][eps]
        for que in ("grupos", "ruido"):
            nombre = f"{que}_{sufijo}"
            if nombre not in vals:
                continue
            v = vals[nombre]
            bien[nombre] = v == b[que]
            if bien[nombre]:
                print(f"✓  {nombre}: bien")
                continue
            hermano_bien = vals.get(f"ruido_{sufijo}") == b["ruido"]   # si el ruido está bien, min_samples y Z también
            if v != int(v):
                print(f"✗  {nombre}: es un conteo, un número entero" +
                      ("; el ruido va como número de clientes, no como proporción" if que == "ruido" else ""))
            elif que == "grupos" and hermano_bien and v == b["grupos"] + 1:
                print(f"✗  {nombre}: estás contando el ruido (-1) como un grupo más")
            elif que == "grupos" and hermano_bien and v == b["grupos"] - 1:
                print(f"✗  {nombre}: te falta sumar 1: las etiquetas empiezan en 0, así que el número de grupos "
                      "es la etiqueta más alta + 1")
            elif v == ref["ms5"][eps][que]:
                print(f"✗  {nombre}: es lo que sale con min_samples por defecto (5); pon min_samples=20")
            elif v == ref["sin_escalar"][eps][que]:
                print(f"✗  {nombre}: es lo que sale sin escalar; usa Z")
            elif que == "grupos" and v == b["grupos"] + 1:
                print(f"✗  {nombre}: estás contando el ruido (-1) como un grupo más")
            elif que == "ruido" and v == 8000 - b["ruido"]:
                print(f"✗  {nombre}: contaste los que NO son ruido; el ruido es la etiqueta -1")
            elif que == "ruido" and v == ref["bien"][0.5 if eps == 0.3 else 0.3]["ruido"]:
                print(f"✗  {nombre}: es el ruido del otro eps; revisa qué etiquetas usaste")
            elif que == "grupos" and v == ref["bien"][0.5 if eps == 0.3 else 0.3]["grupos"]:
                print(f"✗  {nombre}: es el número de grupos del otro eps; revisa qué etiquetas usaste")
            else:
                print(f"✗  {nombre}: no coincide. Pista: el número de grupos es la etiqueta más alta + 1, y el "
                      "ruido, cuántas etiquetas valen -1")
    p = vals.get("pymes_ruido")
    if p is not None:
        b = ref["bien"][0.5]["pymes"]
        if abs(p - b) <= 0.005:
            bien["pymes_ruido"] = True
            print("✓  pymes_ruido: bien")
        elif 1 < p <= 100 and abs(p / 100 - b) <= 0.005:
            bien["pymes_ruido"] = True
            print("✓  pymes_ruido: bien, aunque la escribiste como porcentaje; como proporción va entre 0 y 1")
        else:
            bien["pymes_ruido"] = False
            q = p / 100 if 1 < p <= 100 else p
            if abs(q - ref["bien"][0.5]["pymes_ruido_de_pymes"]) <= 0.005:
                print("✗  pymes_ruido: esa es la proporción de las pymes que quedaron como ruido; la pregunta "
                      "es al revés: de los clientes del ruido, ¿qué proporción son pymes?")
            elif abs(q - ref["bien"][0.3]["pymes"]) <= 0.005:
                print("✗  pymes_ruido: esa es la del ruido con eps 0,3; va la de eps 0,5")
            elif p == int(p) and p > 100:
                print("✗  pymes_ruido: parece un conteo; va la proporción (entre 0 y 1)")
            else:
                print("✗  pymes_ruido: no coincide: filtra a los clientes del ruido (etiquetas_05 == -1) y "
                      "promedia si su tipo_cliente es \"Pyme\"")
    if len(bien) == 5 and all(bien.values()):
        r3, r5 = ref["bien"][0.3], ref["bien"][0.5]
        print(f"\n¡Listo! Con eps 0,5 hay {r5['grupos']} grupo y {r5['ruido']} clientes de ruido, casi todos pymes; "
              f"con 0,3, {r3['grupos']} grupos y {r3['ruido']} de ruido. Mira en «Tus respuestas» qué tienen en "
              "común los clientes de cada grupo de eps 0,3 (la pista está en el enunciado).")


# ---------------------------------------------------------------------------------------------
# Ejercicio 4 (opcional) · Otra columna: solo pistas
# ---------------------------------------------------------------------------------------------

@_seguro
def revisar_otra_columna(*sueltos, columnas=..., grupos5=..., **otros):
    """Ejercicio 4: agrupa con plan_mbps además de las tres columnas, con k = 5. Solo da pistas."""
    import numpy as np

    nombres = ["columnas", "grupos5"]
    if _sobrantes(sueltos, otros, nombres, "revisar_otra_columna(columnas=columnas5, grupos5=grupos5)"):
        return
    d = _datos()
    if d is None:
        return
    if columnas is Ellipsis:
        print("·  columnas: todavía tiene los puntos suspensivos (...) de la plantilla")
        return
    if isinstance(columnas, str):
        columnas = [columnas]
    try:
        lista = list(columnas)
    except TypeError:
        print("✗  columnas: tiene que ser una lista de nombres de columnas")
        return
    if any(c is Ellipsis for c in lista):
        print("·  columnas: todavía tiene los puntos suspensivos (...): cámbialos por \"plan_mbps\", con comillas")
        return
    faltan = [c for c in _COLUMNAS + ["plan_mbps"] if c not in lista]
    sobran = [c for c in lista if c not in _COLUMNAS + ["plan_mbps"]]
    if faltan or sobran:
        print("✗  columnas: van las tres de la clase más \"plan_mbps\"" +
              (f"; faltan {', '.join(map(str, faltan))}" if faltan else "") +
              (f"; sobran {', '.join(map(str, sobran))}" if sobran else ""))
        return
    print("✓  columnas: las cuatro")
    g5 = _etiquetas(grupos5, "grupos5")
    if g5 is None:
        return
    k = len(np.unique(g5))
    if k != 5:
        print(f"✗  grupos5: tiene {_grupos(k)}; el ejercicio pide n_clusters=5")
        return
    from sklearn.preprocessing import StandardScaler
    clientes = d["clientes"]
    datos4 = clientes[_COLUMNAS + ["plan_mbps"]]
    ref = _kmeans(StandardScaler().fit_transform(datos4), 5)
    if _ari(g5, ref) < 0.95:
        if _ari(g5, _kmeans(datos4, 5)) >= 0.95:
            print("✗  grupos5: esos grupos salen sin escalar; escala las cuatro columnas con StandardScaler")
        elif _ari(g5, _kmeans(d["Z"], 5)) >= 0.95:
            print("✗  grupos5: esos grupos salen de las tres columnas de antes (Z); agrupa con Z5, que ya "
                  "trae plan_mbps")
        else:
            print("✗  grupos5: no coinciden con los de la clase: revisa n_init=10 y random_state=2026")
        return
    print("✓  grupos5: 5 grupos, con las cuatro columnas escaladas")
    print("\nPistas para leer la tabla (sin la respuesta):\n"
          "·  Busca la fila de los 4 grupos de antes cuyos clientes se reparten casi por mitades entre dos "
          "columnas (en las otras filas, una sola columna se lleva dos tercios o más).\n"
          "·  Compara el plan_mbps medio de esas dos columnas: "
          "clientes.groupby(grupos5)[\"plan_mbps\"].mean()\n"
          "·  ¿Te sirve ese corte para una oferta? Escríbelo en «Tus respuestas».")
