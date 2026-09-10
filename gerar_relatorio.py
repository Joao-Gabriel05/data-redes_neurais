"""
Monta RELATORIO.html a partir de relatorio_template.html.

Substitui dois tipos de marcador:
  {{FIG:caminho.png}}  -> a imagem embutida como data URI (o artefato publicado
                          nao consegue carregar arquivos locais)
  {{CODE:caminho.py}}  -> o codigo-fonte do arquivo, com HTML escapado
"""

import base64
import html
import re
from pathlib import Path

RAIZ = Path(__file__).parent
TEMPLATE = RAIZ / "relatorio_template.html"
SAIDA = RAIZ / "RELATORIO.html"


def figura(caminho):
    dados = base64.b64encode((RAIZ / caminho).read_bytes()).decode()
    return f"data:image/png;base64,{dados}"


def codigo(caminho):
    return html.escape((RAIZ / caminho).read_text())


if __name__ == "__main__":
    doc = TEMPLATE.read_text()
    for tipo, fn in (("FIG", figura), ("CODE", codigo)):
        doc = re.sub(rf"\{{\{{{tipo}:([^}}]+)\}}\}}",
                     lambda m: fn(m.group(1)), doc)

    faltando = re.findall(r"\{\{[^}]+\}\}", doc)
    if faltando:
        raise SystemExit(f"marcadores nao resolvidos: {faltando}")

    SAIDA.write_text(doc)
    print(f"{SAIDA.name}: {len(doc)/1e6:.2f} MB")
