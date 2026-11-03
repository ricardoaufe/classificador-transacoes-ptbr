# Classificador de Transações Bancárias (PT-BR)

Um modelo que lê a descrição crua de uma transação bancária brasileira e diz em que categoria o gasto se encaixa.

```
"Compra no débito - IFD*CHURRASCO DO GU SE"    →  Alimentação
"Compra no débito - Drogasil2705 Sao Bernardo"  →  Saúde
"Transferência enviada pelo Pix - PESSOA - ..." →  Transferência
```

Comecei este projeto para aprender machine learning na prática, do zero: nunca tinha usado pandas nem scikit-learn. Escolhi um problema real, com dados que eu mesmo coletei, porque quase não existem datasets ou modelos públicos para isso em português.

> 🚧 **Em andamento.** Este README conta o que já foi feito e é atualizado a cada etapa.

---

## Status

| Etapa | Situação |
|---|---|
| Estrutura do repositório e ambiente | ✅ |
| Dataset (coleta, anonimização e rotulagem) | ✅ 667 transações, 4 bancos |
| Exploração dos dados com pandas | ✅ `notebooks/01_exploracao.ipynb` |
| Primeiro modelo (TF-IDF + Regressão Logística) | ✅ F1 macro **0.614** · `notebooks/02_baseline.ipynb` |
| Limpeza do texto | ✅ F1 macro **0.624** · `notebooks/03_limpeza.ipynb` |
| Regras + modelo (sistema híbrido) | ⏳ |
| Dados sintéticos para as categorias pequenas | ⏳ |
| Modelo de linguagem (BERTimbau) | ⏳ |
| Publicação no Hugging Face + demo | ⏳ |

---

## Estrutura

```
classificador-transacoes-ptbr/
├── data/
│   ├── raw/          # extratos originais e planilha de rotulagem (não vão para o GitHub)
│   └── processed/    # descrições anonimizadas e rotuladas
├── notebooks/
│   ├── 01_exploracao.ipynb   # primeiro contato com os dados
│   ├── 02_baseline.ipynb     # primeiro modelo e avaliação
│   └── 03_limpeza.ipynb      # limpeza do texto e ablation
├── src/
│   ├── anonimizar.py         # tira nomes e documentos dos extratos
│   ├── exportar_rotulado.py  # converte a planilha rotulada no CSV final
│   └── limpeza.py            # limpa as descrições antes do modelo
├── requirements.txt
└── README.md
```

---

## 1. Os dados

### De onde vieram
São 667 descrições reais, tiradas dos meus extratos e de faturas de cartão da família:

| Banco | Extrato da conta | Fatura do cartão |
|---|---|---|
| Inter | 238 | 9 |
| Nubank | 50 | – |
| Itaú | – | 318 |
| Bradesco | – | 52 |

A primeira surpresa foi ver como cada banco escreve a mesma loja de um jeito. O Inter corta o nome e cola a cidade logo depois (`Venzol E Frias Restaursao Bernardo Bra`). As faturas cortam os nomes, grudam o número da parcela (`CEDRO COMERCIO ELETR01/03`) e colocam prefixos de maquininha (`MP*`, `IFD*`, `EBN*`). Esse ruído é justamente o que o modelo precisa aprender a ignorar.

### Privacidade
Extrato bancário tem nome, CPF e chave Pix. Antes de qualquer coisa, o script `src/anonimizar.py`:
- guarda **só a descrição** (valor e data são descartados);
- troca nomes de pessoas por `PESSOA`;
- remove CPF, CNPJ, e-mails, telefones e números de conta;
- marca as linhas suspeitas, que revisei uma por uma.

Os arquivos originais ficam em `data/raw/`, que está no `.gitignore` e nunca vai para o repositório.

```bash
python src/anonimizar.py data/raw/extrato.csv --coluna Descrição --fonte conta
python src/exportar_rotulado.py   # valida a planilha e gera data/processed/rotulado_v1.csv
```

### Rotulagem
Rotulei tudo à mão numa planilha, em 11 categorias. Foi mais difícil do que eu esperava: um shopping é estacionamento ou compra? Uma adega é Alimentação ou Lazer? Para ser consistente, criei algumas regras:

- **Rotular pelo que a loja é, não pelo que eu lembro de ter comprado.** O modelo só vê o texto, então minha memória não pode entrar no rótulo.
- **Alimentação** = comida e bebida prontas. **Mercado** = compras de casa (supermercado, hortifruti, açougue).
- **Pix, TED, aplicação e resgate** → Transferência, porque o texto não diz o motivo.
- **Pagamento de fatura, IOF e cashback** → Outros. Pagamento de fatura não é gasto novo: contar de novo seria duplicar.
- **Cobrança no nome do shopping, posto e Tag** → Transporte. **IPTU e condomínio** → Moradia.

| Categoria | Exemplos | % |
|---|---|---|
| Alimentação | 163 | 24,4 |
| Outros | 122 | 18,3 |
| Compras | 104 | 15,6 |
| Lazer | 59 | 8,8 |
| Transferência | 55 | 8,2 |
| Mercado | 53 | 7,9 |
| Transporte | 53 | 7,9 |
| Saúde | 32 | 4,8 |
| Assinaturas | 13 | 1,9 |
| Moradia | 8 | 1,2 |
| Educação | 5 | 0,7 |

Dá para ver que o dataset é **desbalanceado**: tem muito mais Alimentação do que Educação. Isso fez muita diferença na hora de avaliar o modelo.

---

## 2. O primeiro modelo

Para começar, usei uma combinação clássica e simples:
1. **TF-IDF** transforma cada descrição em números, dando mais peso às palavras raras e informativas (`Drogasil`, `Posto`) e menos às que aparecem em tudo (`Compra`, `débito`).
2. **Regressão Logística** aprende quais palavras indicam cada categoria.

A ideia não é ter o melhor modelo ainda, e sim uma **referência** para comparar tudo o que vier depois.

### Como medi
Meu primeiro resultado foi 63% de acurácia, e eu achei ótimo. Depois descobri dois problemas:

- **A acurácia enganava.** O modelo ia bem nas categorias grandes e zerava nas pequenas, e a nota nem percebia. Passei a usar o **F1 macro**, que dá o mesmo peso para todas as categorias.
- **Um número só oscilava.** Adicionar uma única linha ao dataset mudou a nota de 61,2% para 60,4%. Passei a usar **validação cruzada**: o modelo é testado 5 vezes, em partes diferentes dos dados, e eu reporto a média ± a variação.

### Resultados

| Como o texto vira números | Ajuste no modelo | F1 macro | Acurácia |
|---|---|---|---|
| palavras inteiras | nenhum | 0.421 ± 0.032 | 0.594 ± 0.043 |
| pedaços de 2 a 4 letras | nenhum | 0.471 ± 0.031 | 0.640 ± 0.033 |
| palavras inteiras | categorias balanceadas | 0.565 ± 0.064 | 0.610 ± 0.033 |
| **pedaços de 2 a 4 letras** | **categorias balanceadas** | **0.614 ± 0.068** | **0.663 ± 0.033** |

Para comparar: chutar sempre "Alimentação" acertaria 24,6% das vezes.

O que aprendi com cada teste:
- **Pedaços de letras** ajudam porque os bancos cortam os nomes. `Drogasil2705` não bate com `Drogasil` como palavra inteira, mas os pedaços `dro`, `rog`, `gas`... batem.
- **Balancear as categorias** (`class_weight="balanced"`) faz o erro numa categoria pequena pesar tanto quanto numa grande. Foi o maior ganho.
- **Os dois ganhos se somaram quase exatamente**, porque resolvem problemas diferentes. Eu tinha apostado nisso antes de rodar, e deu certo.

### Onde o modelo erra
- **Alimentação virou a "lixeira".** Quando o modelo não sabe, ele chuta Alimentação: das 70 vezes que ele disse Alimentação no teste, só 31 estavam certas. A matriz de confusão mostrou que Mercado, Compras e Transporte iam muito para lá, porque todos têm a mesma cara (`Compra no débito - <loja> <cidade> Bra`).
- **Fatura de cartão é mais difícil que extrato de conta** (52% contra 73% de acerto). A conta tem muito Pix, que é sempre igual. O cartão tem lojas de todo tipo e nomes cortados.
- **Alguns casos nem uma pessoa acertaria.** `CHAPEUZINHO VERMELHO` é uma padaria de bairro, mas nada no texto diz isso.

---

## 3. Limpando o texto

A matriz de confusão mostrou que muita coisa virava Alimentação porque quase toda compra do Inter tem a mesma "moldura": `Compra no débito - <loja> <cidade> Bra`. Escrevi uma função `limpar()` (em `src/limpeza.py`) que tira esse ruído antes do TF-IDF, usando **regex** (expressões regulares, uma minilinguagem para achar padrões em texto):

```
"Compra no débito - Drogasil2705 Sao Bernardo Bra"  →  "drogasil sao bernardo"
"ALLIANZ SEGU*04 de 10"                             →  "allianz segu"
```

A função fica **dentro** do pipeline do modelo, então é aplicada igual no treino, no teste e em qualquer previsão futura.

| Modelo | Sem limpeza | Com limpeza |
|---|---|---|
| palavras inteiras | 0.565 ± 0.064 | **0.612** ± 0.066 |
| pedaços de 2 a 4 letras | 0.614 ± 0.068 | **0.624** ± 0.055 |

O que aprendi:
- **A limpeza ajudou muito o modelo de palavras e quase nada o de letras.** Os pedaços de letras já "enxergavam" `drogasil` dentro de `Drogasil2705`, então já resolviam parte do problema sozinhos. O ganho do modelo de letras ficou dentro da margem, mas a nota ficou mais estável (a variação caiu).
- **Testei uma regra por vez** (desligando cada uma e medindo de novo, o que se chama *ablation*). As que mais ajudaram foram justamente as que tiram `Compra no débito` e `No Estabelecimento`. Nenhuma regra sozinha faz milagre: as quedas individuais somadas (0.046) dão quase exatamente o ganho total (0.047).
- **Metade das regras não mudou nada.** O próprio TF-IDF já ignora pontuação e espaços, e uma regra que apaga números deixava outra sem função. Só descobri isso medindo: lendo o código, todas pareciam úteis.

---

## Próximos passos
1. **Regras + modelo**: uma lista de lojas conhecidas (ex.: Drogasil → Saúde), e o modelo cuida do resto.
2. **Exemplos sintéticos** para Assinaturas, Moradia e Educação, usados só no treino. O teste continua 100% real.
3. **BERTimbau**, um modelo de linguagem treinado em português, para ver o quanto ele supera esta referência.

---

## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Depois, abra os notebooks em `notebooks/` e selecione o kernel `.venv`.

## Tecnologias
Python 3.11 · pandas · scikit-learn · matplotlib · Jupyter · *(em breve)* Hugging Face Transformers
