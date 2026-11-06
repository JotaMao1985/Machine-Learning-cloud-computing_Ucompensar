"""Pasada tipográfica: espacios de no separación en la prosa (no en código, fórmulas ni URL).

    python3 precalculo/espacios.py fuente/diapositivas/sesionN.md              # muestra los cambios
    python3 precalculo/espacios.py fuente/diapositivas/sesionN.md --escribir   # y los guarda

- miles: 12 000 → 12 000 con U+202F (espacio fino de no separación), también en cifras de 7 dígitos
- número + unidad o %: 40,8 % · 5 min · 28,8 °C · 400 mil · 58 000 pesos → U+00A0 (también tras **negrita**)
- operadores entre números (28,8 / 13,9 ≈ 2,07) y «<» de una secuencia → U+00A0 delante

No toca el encabezado, los bloques de código, el código en línea, las fórmulas \\( \\) y $$ $$,
los destinos de los enlaces ni las URL. Es idempotente: una segunda pasada no cambia nada.
Las cifras de cuatro dígitos no se separan (4901), como pide la RAE.
"""
import re
import sys
from pathlib import Path

NB, FINO = " ", " "
PROTEGIDO = re.compile(r"(`[^`\n]*`|\\\(.*?\\\)|\$\$.*?\$\$|\]\([^)]*\)|https?://\S+)", re.S)


def prosa(t):
    t = re.sub(r"(?<![\d.,])\d{1,3}(?:[ \u202f]\d{3})+(?![\d])", lambda m: m.group(0).replace(" ", FINO), t)
    t = re.sub(r"(\d\**) (%|min\b|km\b|°C|h\b|horas\b|mil\b|millones\b|pesos\b)", rf"\1{NB}\2", t)
    t = re.sub(r"(\d\**) (/|≈|=) (\**\d)", rf"\1{NB}\2{NB}\3", t)
    t = re.sub(r" < ", f"{NB}< ", t)
    return t


def pasar(texto):
    cabecera, cuerpo = "", texto
    if texto.startswith("---\n"):
        fin = texto.index("\n---\n", 4) + 5
        cabecera, cuerpo = texto[:fin], texto[fin:]
    salida, en_codigo = [], False
    for linea in cuerpo.split("\n"):
        if linea.startswith("```"):
            en_codigo = not en_codigo
            salida.append(linea)
            continue
        if en_codigo or linea.startswith("<!--") or linea.startswith("  "):
            salida.append(linea)
            continue
        partes = PROTEGIDO.split(linea)
        salida.append("".join(p if i % 2 else prosa(p) for i, p in enumerate(partes)))
    return cabecera + "\n".join(salida)


if __name__ == "__main__":
    ruta = Path(sys.argv[1])
    viejo = ruta.read_text(encoding="utf-8")
    nuevo = pasar(viejo)
    cambios = 0
    for a, b in zip(viejo.split("\n"), nuevo.split("\n")):
        if a != b:
            cambios += 1
            print("·", b.replace(NB, "⍽").replace(FINO, "ˑ")[:160])
    print(f"{cambios} líneas cambiadas" + (" (escritas)" if cambios and "--escribir" in sys.argv else ""))
    if "--escribir" in sys.argv and cambios:
        ruta.write_text(nuevo, encoding="utf-8")
