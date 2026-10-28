"""
Anonimiza extratos bancários exportados em CSV e gera uma planilha pronta para rotular.

Uso:
    python anonimizar.py data/raw/nubank_conta.csv --coluna Descrição --fonte conta
    python anonimizar.py data/raw/nubank_cartao.csv --coluna title --fonte cartao
    python anonimizar.py data/raw/inter.csv --coluna Histórico Descrição --fonte conta

Saída: data/processed/para_rotular.csv (acrescenta linhas se o arquivo já existir)
"""
import argparse
import os
import re

import pandas as pd

# Cada item: (padrão regex, texto que entra no lugar)
SUBSTITUICOES = [
    # Nome depois de "Pix - " ou "Transferência ... - " (formato da conta Nubank)
    (r"(?i)((?:pix|transfer[êe]ncia)[^-]*\s-\s)[^-]+", r"\1PESSOA "),
    # CPF completo ou mascarado: 123.456.789-00, •••.456.789-••
    (r"[\d•*]{3}\.[\d•*]{3}\.[\d•*]{3}-[\d•*]{2}", " "),
    # CNPJ: 12.345.678/0001-90
    (r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", " "),
    # E-mail (chave PIX)
    (r"\S+@\S+", " "),
    # Agência e conta
    (r"(?i)ag[êe]ncia:?\s*\S+", " "),
    (r"(?i)conta:?\s*\S+", " "),
    # Sequências longas de números (telefone, cartão, código de autorização)
    (r"\d{6,}", " "),
    # Traços que ficaram sobrando: "- -" vira "-"
    (r"(\s-\s*)+-", " -"),
    # Espaços repetidos
    (r"\s+", " "),
]

# Linhas com estas palavras podem ter nome de pessoa: revise à mão
SUSPEITAS = r"(?i)\b(?:pix|transf|ted|doc|enviad|recebid)"


def anonimizar(texto: str) -> str:
    for padrao, troca in SUBSTITUICOES:
        texto = re.sub(padrao, troca, texto)
    return texto.strip(" -")


def ler_csv(caminho: str, coluna: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(caminho, encoding=encoding) as f:
                linhas = f.readlines()
        except UnicodeDecodeError:
            continue
        # Alguns bancos (ex.: Inter) põem linhas de resumo antes da tabela.
        # Pulamos tudo até a linha que contém o nome da coluna.
        inicio = next((i for i, l in enumerate(linhas) if coluna in l), 0)
        sep = ";" if linhas[inicio].count(";") > linhas[inicio].count(",") else ","
        return pd.read_csv(caminho, sep=sep, skiprows=inicio, encoding=encoding)
    raise ValueError(f"Não consegui ler {caminho}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arquivo")
    parser.add_argument("--coluna", required=True, nargs="+",
                        help="coluna(s) com a descrição; se forem várias, são juntadas com ' - '")
    parser.add_argument("--fonte", required=True, choices=["conta", "cartao"])
    parser.add_argument("--saida", default="data/processed/para_rotular.csv")
    args = parser.parse_args()

    df = ler_csv(args.arquivo, args.coluna[0])
    print("Colunas encontradas:", list(df.columns))

    # Fica só com a descrição; valor e data são descartados aqui
    saida = pd.DataFrame()
    texto = df[args.coluna].fillna("").astype(str).agg(" - ".join, axis=1)
    saida["descricao"] = texto.map(anonimizar)
    saida["categoria"] = ""
    saida["fonte"] = args.fonte
    saida["revisar"] = saida["descricao"].str.contains(SUSPEITAS).map({True: "SIM", False: ""})

    if os.path.exists(args.saida):
        saida = pd.concat([pd.read_csv(args.saida, keep_default_na=False), saida])
    saida = saida.drop_duplicates(subset="descricao")

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    saida.to_csv(args.saida, index=False, encoding="utf-8")
    print(f"{len(saida)} descrições únicas salvas em {args.saida}")
    print(f"{(saida['revisar'] == 'SIM').sum()} marcadas para revisão manual")


if __name__ == "__main__":
    main()
