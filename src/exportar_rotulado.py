"""
Converte a planilha de trabalho (data/raw/rotulagem.xlsx) no dataset oficial
(data/processed/rotulado_v1.csv), em UTF-8 e separado por vírgula.

Uso:
    python src/exportar_rotulado.py
"""
import pandas as pd

ENTRADA = "data/raw/rotulagem.xlsx"
SAIDA = "data/processed/rotulado_v1.csv"

CATEGORIAS = {
    "Alimentação", "Transporte", "Mercado", "Moradia", "Saúde", "Assinaturas",
    "Compras", "Lazer", "Educação", "Transferência", "Outros",
}

df = pd.read_excel(ENTRADA)
df.columns = df.columns.str.strip().str.lower()   # DESCRICAO -> descricao
df = df[["descricao", "categoria", "fonte", "banco"]]

# Tira espaços sobrando: "Saúde " viraria uma categoria diferente de "Saúde"
for coluna in ["descricao", "categoria", "fonte", "banco"]:
    df[coluna] = df[coluna].astype("string").str.strip().replace("", pd.NA)
df = df.dropna(subset=["descricao"])               # linhas sem descrição

# Mesma descrição com rótulos diferentes: avisa antes de ficar só com a primeira
rotulos = df.dropna(subset=["categoria"]).groupby("descricao")["categoria"].nunique()
conflitos = rotulos[rotulos > 1].index
if len(conflitos):
    print("⚠ Mesma descrição com categorias diferentes (fica a primeira):")
    print(df[df["descricao"].isin(conflitos)][["descricao", "categoria"]].to_string())
df = df.drop_duplicates(subset="descricao")

# Avisos (não impedem a exportação)
invalidas = df[df["categoria"].notna() & ~df["categoria"].isin(CATEGORIAS)]
if len(invalidas):
    print("⚠ Categorias fora da lista:")
    print(invalidas[["descricao", "categoria"]].to_string())

sem_fonte_banco = df["fonte"].isna() | df["banco"].isna()
if sem_fonte_banco.any():
    print(f"⚠ {sem_fonte_banco.sum()} linhas sem fonte ou banco")

em_branco = df["categoria"].isna().sum()
df.to_csv(SAIDA, index=False, encoding="utf-8")

print(f"{len(df)} linhas salvas em {SAIDA} ({em_branco} ainda sem categoria)")
print(df["categoria"].value_counts().to_string())
