# Classificador de Transações Bancárias (PT-BR)

Modelo que recebe a descrição crua de uma transação bancária brasileira e devolve a categoria do gasto.

```
"Compra no débito - IFD*CHURRASCO DO GU SE"    →  Alimentação
"Compra no débito - Drogasil2705 Sao Bernardo"  →  Saúde
"Transferência enviada pelo Pix - PESSOA - ..." →  Transferência
```

Quase não existem modelos ou datasets públicos para esse problema em português. Este projeto constrói os dois do zero: coleta e rotulagem dos dados, modelo, avaliação e publicação.

> 🚧 **Em andamento.** Este README descreve só o que já foi feito e é atualizado a cada etapa.

---

## Status

| Etapa | Situação |
|---|---|
| Estrutura do repositório e ambiente | ✅ |
| Dataset v1 (coleta, anonimização e rotulagem) | ✅ 667 transações, 4 bancos |
| Exploração dos dados (pandas) | ✅ `notebooks/01_exploracao.ipynb` |
| Baseline (TF-IDF + Regressão Logística) | ✅ F1 macro **0.614** · `notebooks/02_baseline.ipynb` |
| Limpeza do texto e regras | ⏳ |
| Dados sintéticos para categorias pequenas | ⏳ |
| Fine-tuning (BERTimbau) | ⏳ |
| Publicação no Hugging Face + demo | ⏳ |

---

## Estrutura

```
classificador-transacoes-ptbr/
├── data/
│   ├── raw/          # extratos originais e planilha de rotulagem (não versionado)
│   └── processed/    # descrições anonimizadas e rotuladas
├── notebooks/
│   ├── 01_exploracao.ipynb   # análise exploratória com pandas
│   └── 02_baseline.ipynb     # TF-IDF + Regressão Logística e avaliação
├── src/
│   ├── anonimizar.py         # extrato bruto → descrições anonimizadas
│   └── exportar_rotulado.py  # planilha de rotulagem → CSV oficial, com validações
├── requirements.txt
└── README.md
```

---

## Dataset

### Fontes
667 descrições reais e únicas, de quatro bancos e dois tipos de documento:

| Banco | Extrato da conta (`conta`) | Fatura do cartão (`cartao`) |
|---|---|---|
| Inter | 238 | 9 |
| Nubank | 50 | – |
| Itaú | – | 318 |
| Bradesco | – | 52 |

Cada banco escreve a mesma loja de um jeito diferente. O Inter corta o nome do estabelecimento e cola a cidade logo depois (`Venzol E Frias Restaursao Bernardo Bra`). As faturas de cartão trazem nomes cortados, parcelas grudadas (`CEDRO COMERCIO ELETR01/03`) e prefixos de adquirente (`MP*`, `IFD*`, `EBN*`). Essa variedade é proposital: é o ruído que o modelo precisa aprender a ignorar.

### Formato
`data/processed/rotulado_v1.csv` (UTF-8, separado por vírgula):

| coluna | descrição |
|---|---|
| `descricao` | texto **bruto**, como veio do banco |
| `categoria` | rótulo atribuído à mão |
| `fonte` | `conta` (extrato) ou `cartao` (fatura) |
| `banco` | banco de origem, para análise de erros por fonte |

### Categorias
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

O dataset é **desbalanceado**: as três maiores categorias somam quase 60% das linhas, enquanto Assinaturas, Moradia e Educação têm menos de 15 exemplos cada. Isso define a métrica principal do projeto (F1 macro, abaixo).

### Privacidade
O script `src/anonimizar.py` lê o extrato exportado e:
- mantém **só a descrição** (valor e data são descartados);
- troca nomes de pessoas em Pix e transferências por `PESSOA`;
- remove CPF, CNPJ, e-mails, telefones, agência, conta e sequências longas de números;
- marca as linhas suspeitas para **revisão manual**, que foi feita em todas.

Nas faturas, compras feitas diretamente com pessoas físicas também tiveram o nome trocado por `PESSOA`. Os extratos originais ficam em `data/raw/`, que está no `.gitignore` e nunca vai para o repositório.

```bash
python src/anonimizar.py data/raw/extrato.csv --coluna Descrição --fonte conta --saida data/processed/para_rotular.csv
python src/exportar_rotulado.py   # valida a planilha e gera data/processed/rotulado_v1.csv
```

### Convenções de rotulagem
Rotular uma transação é mais ambíguo do que parece. As regras abaixo garantem que textos iguais sempre recebam o mesmo rótulo:

- **Rotular pelo que o estabelecimento é, não pela compra específica.** O modelo só vê o texto, então a memória de quem rotulou não pode entrar no rótulo.
- **Alimentação** = comida e bebida prontas (restaurante, padaria, bar, adega). **Mercado** = ingredientes e compras de casa (supermercado, atacado, hortifruti, açougue, conveniência).
- **Pix, TED, aplicação e resgate** → Transferência. O texto não diz o motivo, só que o dinheiro mudou de conta.
- **Pagamento de fatura, IOF, cashback e doações** → Outros. Pagamento de fatura não é gasto novo: contá-lo seria duplicar.
- **Cobrança no nome do próprio shopping, posto, pedágio e Tag** → Transporte.
- **Prefeitura (IPTU) e condomínio** → Moradia.
- **Academia e farmácia** → Saúde. **Viagem, hotel e ingressos** → Lazer. **Software e streaming** → Assinaturas.
- **Educação** só para escola, faculdade e cursos (livraria vai em Compras).

---

## Baseline: TF-IDF + Regressão Logística

O primeiro modelo é propositalmente simples: transforma cada descrição em números com **TF-IDF** e classifica com **Regressão Logística**. Ele serve de referência para tudo que vier depois.

### Métrica
A métrica principal é o **F1 macro**: a média do F1 de cada categoria, todas com o mesmo peso. A acurácia sozinha engana neste dataset, porque um modelo que acerta bem as categorias grandes já tem nota alta mesmo zerando as pequenas.

Todos os números abaixo vêm de **validação cruzada estratificada com 5 partes** (média ± desvio padrão), que é mais confiável que um único sorteio de treino/teste num dataset deste tamanho.

### Resultados

| Representação do texto | Regressão Logística | F1 macro | Acurácia |
|---|---|---|---|
| palavras | padrão | 0.421 ± 0.032 | 0.594 ± 0.043 |
| n-gramas de caracteres (2–4) | padrão | 0.471 ± 0.031 | 0.640 ± 0.033 |
| palavras | `class_weight="balanced"` | 0.565 ± 0.064 | 0.610 ± 0.033 |
| **n-gramas de caracteres (2–4)** | **`class_weight="balanced"`** | **0.614 ± 0.068** | **0.663 ± 0.033** |

Referência: chutar sempre a categoria mais comum (Alimentação) dá acurácia de 0.246.

- **N-gramas de caracteres** ajudam porque os bancos cortam e colam palavras (`Drogasil2705`, `Restaursao`). Pedaços de 2 a 4 letras ainda batem com a forma completa, enquanto a palavra inteira não bate.
- **`class_weight="balanced"`** faz o erro numa categoria pequena pesar tanto quanto o erro numa grande. Foi o maior ganho isolado (+14 pontos de F1 macro), sem perder acurácia.
- Os dois ganhos **praticamente se somam** (0.421 → 0.614), porque atacam problemas diferentes: texto truncado e desbalanceamento.

### Análise de erros (modelo por palavras, sem balanceamento)
- **Alimentação vira a "categoria lixeira".** Ela tem recall de 0.94, mas precisão de 0.44: das 70 previsões de Alimentação no teste, só 31 estavam certas. Compras, Outros, Mercado e Transporte são os mais confundidos com ela, porque compartilham o mesmo formato (`Compra no débito - <loja> <cidade> Bra`).
- **Mercado, Transporte e Saúde** têm precisão de 1.00 e recall baixo: quando o modelo escolhe essas categorias ele acerta, mas quase nunca as escolhe. É exatamente o que o balanceamento corrige.
- **Transferência acerta 100%**, porque o padrão `Pix - PESSOA - <banco>` é sempre igual.
- **Fatura de cartão é mais difícil que extrato de conta** (52% contra 73% de acerto): o cartão tem muito mais variedade de lojas e nomes cortados, enquanto a conta é dominada por Pix.
- Alguns casos são impossíveis de deduzir só pelo texto, mesmo para uma pessoa (ex.: `CHAPEUZINHO VERMELHO` é uma padaria; `A B & B Comercio De Pr...` é uma razão social genérica).

### Próximos passos
1. **Limpeza do texto**: remover prefixos fixos ("Compra no débito - ", "Bra") e separar números colados aos nomes, aplicada no treino e na previsão.
2. **Regras + modelo**: dicionário para lojas conhecidas (ex.: Drogasil → Saúde), com o modelo cobrindo o resto.
3. **Dados sintéticos** para Assinaturas, Moradia e Educação, usados só no treino (o teste continua 100% real).
4. **Fine-tuning do BERTimbau**, comparado com este baseline.

---

## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Depois, abra os notebooks em `notebooks/` e selecione o kernel `.venv`.

---

## Tecnologias
Python 3.11 · pandas · scikit-learn · matplotlib · Jupyter · *(em breve)* Hugging Face Transformers
