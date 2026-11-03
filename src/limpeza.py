import re

def limpar(texto: str) -> str:
    t = texto.lower()

    # Prefixo extrato Inter
    t = re.sub(r"compra no d[ée]bito\s*-\s*", " ", t)

    # Sufixo "Bra"
    t = re.sub(r"\s+bra$", " ", t)

    # Prefixo "no estabelecimento"
    t = re.sub(r"no estabelecimento\s*\s*", " ", t)

    # 05/10
    t = re.sub(r"\d{2}/\d{2}", " ", t)

    # 05 de 10
    t = re.sub(r"\d+ de \d+", " ", t)

    # Sufixo "-ct" do Itaú
    t = re.sub(r"-ct\b", "", t)

    # Separa letras de números colados
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)

    # Remove os números separados
    t = re.sub(r"\d+", " ", t)

    # Remove qualquer coisa que não seja letra nem número
    t = re.sub(r"[^\w\s]", " ", t)

    # Espaços repetidos viram um só
    t = re.sub(r"\s+", " ", t)
    return t.strip()