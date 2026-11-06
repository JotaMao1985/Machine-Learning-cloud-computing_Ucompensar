"""Páginas fijas del curso: fuente/<página> → Htmls/<página>.

    python precalculo/portada.py

Páginas: index.html (la portada) y proyecto.html (el enunciado del proyecto final), si ya
existe su fuente.

Marcadores:
    {{LOGO}}       logo institucional como data URI (fuente/comun/logo-ucompensar.png)
    {{COLAB:N}}    enlace que abre en Colab el notebook de la sesión N publicado en GitHub
    {{DATOS_URL}}  enlace de descarga de datos/clientes.csv en GitHub
Si queda un marcador sin resolver, se detiene con error.
"""

import base64
import html
import re
from pathlib import Path

from notebook import SESIONES

RAIZ = Path(__file__).resolve().parent.parent
REPO = "JotaMao1985/Machine-Learning-cloud-computing_Ucompensar"
PAGINAS = "https://jotamao1985.github.io/Machine-Learning-cloud-computing_Ucompensar/"
DATOS_URL = f"https://raw.githubusercontent.com/{REPO}/main/datos/clientes.csv"
# La portada es obligatoria; las demás se construyen solo si existe su fuente
FIJAS = ["index.html", "proyecto.html"]


def enlace_colab(numero):
    return f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{SESIONES[numero]}"


def construir(nombre, logo):
    texto = (RAIZ / "fuente" / nombre).read_text(encoding="utf-8")
    texto = texto.replace("{{LOGO}}", f"data:image/png;base64,{logo}")
    texto = texto.replace("{{DATOS_URL}}", html.escape(DATOS_URL))
    texto = re.sub(r"\{\{COLAB:(\d+)\}\}", lambda m: html.escape(enlace_colab(int(m.group(1)))), texto)
    restos = re.findall(r"\{\{[^}]*\}\}", texto)
    if restos:
        raise SystemExit(f"Marcadores sin resolver en {nombre}: {restos}")
    destino = RAIZ / "Htmls" / nombre
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print("Página:", destino.relative_to(RAIZ))


def main():
    logo = base64.b64encode((RAIZ / "fuente/comun/logo-ucompensar.png").read_bytes()).decode()
    for nombre in FIJAS:
        if nombre == "index.html" or (RAIZ / "fuente" / nombre).exists():
            construir(nombre, logo)


if __name__ == "__main__":
    main()
