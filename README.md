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
| Dataset (coleta, anonimização e rotulagem) | ✅ 667 transações reais, 4 bancos |
| Exploração dos dados com pandas | ✅ `notebooks/01_exploracao.ipynb` |
| Primeiro modelo (TF-IDF + Regressão Logística) | ✅ F1 macro **0.638** · `notebooks/02_baseline.ipynb` |
| Limpeza do texto | ✅ F1 macro **0.644** · `notebooks/03_limpeza.ipynb` |
| Dados extras (web e sintéticos) | ✅ testado: **não ajudou** · `notebooks/04_dados_extras.ipynb` |
| Regras + modelo (sistema híbrido) | ⏳ |
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
│   ├── 03_limpeza.ipynb      # limpeza do texto e ablation
│   └── 04_dados_extras.ipynb # dados da web e sintéticos: ajudam ou atrapalham?
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
| Compras | 107 | 16,0 |
| Lazer | 59 | 8,8 |
| Transferência | 55 | 8,2 |
| Mercado | 53 | 7,9 |
| Transporte | 53 | 7,9 |
| Saúde | 32 | 4,8 |
| Assinaturas | 13 | 1,9 |
| Educação | 5 | 0,7 |
| Moradia | 5 | 0,7 |

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
| palavras inteiras | nenhum | 0.421 ± 0.022 | 0.601 ± 0.037 |
| pedaços de 2 a 4 letras | nenhum | 0.472 ± 0.024 | 0.645 ± 0.025 |
| palavras inteiras | categorias balanceadas | 0.581 ± 0.040 | 0.618 ± 0.026 |
| **pedaços de 2 a 4 letras** | **categorias balanceadas** | **0.638 ± 0.095** | **0.664 ± 0.049** |

Todos os números desta seção e da próxima usam só os 667 exemplos reais, com validação cruzada de 5 partes. Eles mudaram um pouco desde a primeira versão deste README porque revisei alguns rótulos.

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
| palavras inteiras | 0.581 ± 0.040 | **0.614** ± 0.046 |
| pedaços de 2 a 4 letras | 0.638 ± 0.095 | **0.644** ± 0.084 |

O que aprendi:
- **A limpeza ajudou muito o modelo de palavras e quase nada o de letras.** Os pedaços de letras já "enxergavam" `drogasil` dentro de `Drogasil2705`, então já resolviam parte do problema sozinhos. O ganho do modelo de letras ficou dentro da margem, mas a nota ficou mais estável (a variação caiu).
- **Testei uma regra por vez** (*ablation*). As que mais ajudaram foram justamente as que tiram `No Estabelecimento` (−0.014 sem ela) e `Compra no débito` (−0.006). Nenhuma regra sozinha faz milagre: o ganho vem de várias regras pequenas somadas.
- **Metade das regras não mudou nada.** O próprio TF-IDF já ignora pontuação e espaços, e uma regra que apaga números deixava outra sem função. Só descobri isso medindo: lendo o código, todas pareciam úteis.

---

## 4. Dados extras: por que os sintéticos não ajudaram

Três categorias têm pouquíssimos exemplos reais: Assinaturas (13), Educação (5) e Moradia (5). A ideia parecia óbvia: dar mais exemplos dessas categorias para o modelo. Juntei dois tipos de dado extra:

- **web** (11 linhas): descrições reais que pessoas publicaram na internet, em sites de reclamação e fóruns;
- **sintéticos** (66 linhas): descrições que eu escrevi, com ajuda de um chatbot, imitando o jeito de cada banco (`DM *SPOTIFYBR`, `Pagamento de boleto efetuado - SABESP`).

### Como medi sem me enganar
Os extras entram **só no treino**. O teste continua 100% com os meus dados reais, sempre nas mesmas partes (os mesmos *folds*). Se os sintéticos entrassem no teste, a nota mediria se o modelo acerta exemplos que eu inventei, e não os meus extratos de verdade. Para isso, escrevi o loop da validação cruzada à mão em vez de usar a função pronta do scikit-learn.

Nesta etapa passei a usar 10 partes em vez de 5 e revisei alguns rótulos, por isso a referência "só real" mudou em relação à seção 3.

### Resultados (pedaços de letras + limpeza, 10 folds)

| Treino | Extras | F1 macro |
|---|---|---|
| **só dados reais** | 0 | **0.664 ± 0.088** |
| reais + web | 11 | 0.646 ± 0.065 |
| reais + todos os sintéticos | 66 | 0.646 ± 0.080 |
| reais + sintéticos filtrados | 39 | 0.658 ± 0.091 |

### O que aprendi
- **Mais dados não é o mesmo que dados melhores.** Os sintéticos deixaram o modelo um pouco **pior**.
- **A análise de erros mostrou o porquê.** Na primeira versão, `UBER *ONE` (assinatura) fez corridas reais de Uber virarem Assinaturas. E como meus sintéticos do Itaú eram quase todos assinaturas em letras maiúsculas, o modelo aprendeu que "nome curto em maiúsculas = Assinaturas", e uma padaria (`CHARLOT`) virou assinatura. Isso tem nome: *shortcut learning*, quando o modelo aprende uma pista que funciona no treino mas não vale no mundo real.
- **Filtrei no código, não no arquivo.** Escrevi regras que marcam os sintéticos ambíguos (intermediários de pagamento como `EBANX*` e `MP*`, que aparecem em várias categorias, e o formato curto do Itaú), sem apagar nada do dataset. Assim dá para desfazer e comparar.
- **O filtro recuperou quase toda a perda, mas não trouxe ganho.** Filtrar ainda mais (tirando também combinações de banco e categoria que não existem nos dados reais) voltou a piorar. E todas as diferenças ficaram dentro da margem (±), então o resultado honesto é um empate.
- **Decisão:** o modelo oficial continua treinado **só com dados reais**. A curva de aprendizado (a nota conforme o treino cresce) ainda está subindo, então o melhor investimento é coletar mais dado real.

---

## Próximos passos
1. **Regras + modelo**: uma lista de lojas conhecidas (ex.: Drogasil → Saúde), e o modelo cuida do resto.
2. **BERTimbau**, um modelo de linguagem treinado em português, para ver o quanto ele supera esta referência.
3. **Publicar no Hugging Face** o dataset e o modelo, com uma demo.

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
