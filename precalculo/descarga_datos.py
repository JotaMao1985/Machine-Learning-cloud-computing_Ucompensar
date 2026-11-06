"""Datos del caso «Yarumo Fibra S.A.S.» (empresa FICTICIA), descargados del repositorio del curso.

Deja en datos/clientes.csv los 8000 clientes del curso y comprueba su huella SHA-256: si el
archivo no es idéntico al del curso, no se usa. Solo usa la biblioteca estándar de Python.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

DATOS_URL = "https://raw.githubusercontent.com/JotaMao1985/Machine-Learning-cloud-computing_Ucompensar/main/datos/clientes.csv"
DATOS_SHA256 = "2d4fad61a49efcf1d071b6c412e5224e71eca23047cbbac6b4017c8627f9c93b"
PORTADA = "https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/"


def _es_del_curso(contenido):
    return hashlib.sha256(contenido).hexdigest() == DATOS_SHA256


def _sin_datos(motivo):
    return (f"No pude dejar listos los datos: {motivo}.\n"
            f"Descarga clientes.csv desde la portada del curso ({PORTADA}), súbelo a Colab con el "
            "ícono de carpeta de la izquierda (Subir al almacenamiento de la sesión) y vuelve a "
            "ejecutar esta celda.")


def _leer_url():
    # Solo al preparar el material: la variable apunta a una copia local del mismo archivo
    local = os.environ.get("YARUMO_CSV_LOCAL")
    if local:
        return Path(local).read_bytes()
    with urllib.request.urlopen(DATOS_URL, timeout=30) as respuesta:
        return respuesta.read()


def descargar_datos(carpeta="datos"):
    """Deja <carpeta>/clientes.csv, idéntico al del curso, y devuelve su ruta."""
    ruta = Path(carpeta) / "clientes.csv"
    subido = Path("clientes.csv")  # por si se subió a mano a la carpeta de trabajo de Colab
    if ruta.exists() and _es_del_curso(ruta.read_bytes()):
        contenido, origen = ruta.read_bytes(), "ya estaban"
    elif subido.exists() and _es_del_curso(subido.read_bytes()):
        contenido, origen = subido.read_bytes(), "copiados del archivo que subiste"
    else:
        try:
            contenido = _leer_url()
        except OSError as error:
            raise RuntimeError(_sin_datos(f"no se pudo descargar el archivo ({error})")) from None
        if not _es_del_curso(contenido):
            raise RuntimeError(_sin_datos("el archivo descargado no es el del curso"))
        origen = "descargados del repositorio del curso"
    if origen != "ya estaban":
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_suffix(".tmp")  # sin archivos a medias si algo falla al escribir
        temporal.write_bytes(contenido)
        temporal.replace(ruta)
    lineas = contenido.decode("utf-8").splitlines()
    print(f"Listo: {ruta} con {len(lineas) - 1} clientes y {len(lineas[0].split(','))} columnas, {origen}")
    return ruta


if __name__ == "__main__":
    import sys
    descargar_datos(sys.argv[1] if len(sys.argv) > 1 else "datos")
