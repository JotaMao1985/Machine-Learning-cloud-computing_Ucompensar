"""Comprueba que el código y las salidas proyectadas coinciden con el notebook y su ejecución.

    python precalculo/verifica_diapositivas.py 1

Para cada bloque ```python [id] de fuente/diapositivas/sesionN.md:

- salida: compara sus líneas #> con precalculo/salidas_sesionN.json (sin espacios al final).
  Una salida partida en dos líneas en la diapositiva se acepta si, unida, coincide;
- código: cada línea proyectada tiene que estar en la celda [id] de celdas_sesionN.py, en el
  mismo orden (se pueden quitar líneas y comentarios, no cambiar código);
- ancho: ninguna línea proyectada pasa de 76 caracteres.
"""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "precalculo"))
from notebook import leer_celdas  # noqa: E402

ANCHO = 76


def normal(texto):
    return re.sub(r"\s+", " ", texto).strip()


def sin_comentario(linea):
    return re.sub(r"\s+#(?!>).*$", "", linea).rstrip()


def codigo_en_celda(proyectado, celda):
    """¿Las líneas proyectadas son una subsecuencia de las de la celda (con o sin comentario)?"""
    lineas = [l.rstrip() for l in celda.split("\n")]
    i = 0
    for linea in proyectado:
        while i < len(lineas) and linea not in (lineas[i], sin_comentario(lineas[i])):
            i += 1
        if i == len(lineas):
            return linea
        i += 1
    return None


def main(n):
    salidas = json.loads((RAIZ / f"precalculo/salidas_sesion{n}.json").read_text(encoding="utf-8"))
    fuente = (RAIZ / f"fuente/diapositivas/sesion{n}.md").read_text(encoding="utf-8")
    celdas = {c["id"]: c["fuente"] for c in leer_celdas(RAIZ / f"precalculo/celdas_sesion{n}.py")
              if c["tipo"] == "code"}
    errores = 0
    # un bloque de código sin [id] (o con la cabecera mal escrita) no se podría verificar
    for cabecera in re.findall(r"^[ \t]*```[ \t]*(?:python|py|ipython)\b[^\n]*$", fuente, re.M):
        if not re.match(r"[ \t]*```python \[\w+\]", cabecera):
            errores += 1
            print(f"✗ bloque sin [id] al comienzo, no se puede verificar: {cabecera.strip()}")
    for ident, cuerpo in re.findall(r"```python \[(\w+)\][^\n]*\n(.*?)```", fuente, re.S):
        lineas = cuerpo.rstrip("\n").split("\n")
        problemas = []
        largas = [l for l in lineas if len(l) > ANCHO]
        if largas:
            problemas.append(f"{len(largas)} línea(s) de más de {ANCHO} caracteres: {largas[0][:40]}…")
        if ident not in celdas:
            problemas.append(f"no hay celda [{ident}] en celdas_sesion{n}.py")
        else:
            codigo = [l.rstrip() for l in lineas if l.strip() and not l.startswith("#>")]
            fuera = codigo_en_celda(codigo, celdas[ident])
            if fuera is not None:
                problemas.append(f"código que no está en la celda (o fuera de orden): {fuera}")
        proyectada = [l[2:] for l in lineas if l.startswith("#>")]
        if proyectada:
            real = salidas.get(ident)
            if real is None:
                problemas.append(f"no está en salidas_sesion{n}.json")
            elif normal(" ".join(proyectada)) != normal(real):
                problemas.append(f"la salida proyectada no coincide\n  diapositiva: "
                                 f"{normal(' '.join(proyectada))}\n  real:        {normal(real)}")
        if problemas:
            errores += 1
            print(f"✗ [{ident}]: " + "\n  ".join(problemas))
        else:
            print(f"✓ [{ident}]" + (" (código y salida)" if proyectada else " (código)"))
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
