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
| Dataset v1 (coleta, anonimização e rotulagem) | ✅ 138 transações |
| Exploração dos dados (pandas) | ✅ `notebooks/01_exploracao.ipynb` |
| Ampliar dataset para ~300 transações | ⏳ |
| A definir.....                       |     |

 
---
 
## Estrutura
 
```
classificador-transacoes-ptbr/
├── data/
│   ├── raw/          # extratos originais (não versionado)
│   └── processed/    # descrições anonimizadas e rotuladas
├── notebooks/        # exploração
├── src/              # código (anonimização, futuramente limpeza, regras e treino)
├── requirements.txt
└── README.md
```
 
---
 
## Dataset
 
### Fontes
Descrições reais exportadas de três bancos, em dois tipos de documento:
 
| Banco | Documento | `fonte` |
|---|---|---|
| Inter | extrato da conta | `conta` |
| Nubank | extrato da conta (jul–out/2026) | `conta` |
| Bradesco | fatura do cartão de crédito | `cartao` |
 
Cada banco escreve a mesma loja de um jeito diferente. O Inter, por exemplo, corta o nome do estabelecimento e cola a cidade logo depois (`Venzol E Frias Restaursao Bernardo Bra`). A fatura do cartão traz nomes cortados, parcelas grudadas (`CEDRO COMERCIO ELETR01/03`) e prefixos de adquirente (`MP*`, `IFD*`, `CAPPTA *`). Essa variedade é proposital: é o ruído que o modelo precisa aprender a ignorar.
 
### Formato
`data/processed/rotulado_v1.csv` (UTF-8, separado por vírgula):
 
| coluna | descrição |
|---|---|
| `descricao` | texto **bruto**, como veio do banco |
| `categoria` | rótulo atribuído à mão |
| `fonte` | `conta` (extrato) ou `cartao` (fatura) |
| `banco` | banco de origem, para análise de erros por fonte |
 
### Categorias
Alimentação, Transporte, Mercado, Moradia, Saúde, Assinaturas, Compras, Lazer, Educação, Transferência, Outros.
 
### Privacidade
O script `src/anonimizar.py` lê o extrato exportado e:
- mantém **só a descrição** (valor e data são descartados);
- troca nomes de pessoas em Pix e transferências por `PESSOA`;
- remove CPF, CNPJ, e-mails, agência, conta e sequências longas de números;
- marca as linhas suspeitas para **revisão manual**, que foi feita em todas.
Os extratos originais ficam em `data/raw/`, que está no `.gitignore` e nunca vai para o repositório.
 
```bash
python src/anonimizar.py data/raw/extrato.csv --coluna Descrição --fonte conta --saida data/processed/para_rotular.csv
```
 
### Convenções de rotulagem
Rotular uma transação é mais ambíguo do que parece. As regras abaixo garantem que textos iguais sempre recebam o mesmo rótulo:
 
- **Rotular pelo que o estabelecimento costuma ser, não pela compra específica.** O modelo só vê o texto, então a memória de quem rotulou não pode entrar no rótulo.
- **Alimentação** = comida pronta para comer. **Mercado** = ingredientes e compras de casa (hortifruti, açougue, empório, conveniência).
- **Pix, TED, aplicação e resgate** → Transferência. O texto não diz o motivo, só que o dinheiro mudou de conta.
- **Pagamento de fatura** → Outros, porque os gastos já estão dentro da fatura e contá-los de novo seria duplicar.
- **Cobrança no nome do próprio shopping** → Transporte (estacionamento).
- **Boleto de prefeitura** → Moradia (IPTU).
- **Doações** → Outros. **Educação** só para escola, faculdade e cursos (livraria vai em Compras).
### Observações da exploração
- O dataset está **desbalanceado**: Transferência é de longe a maior categoria, e quase todas essas linhas seguem o padrão `Pix - PESSOA - <banco>`. Lazer, Assinaturas e Educação têm poucos exemplos. A próxima coleta prioriza **faturas de cartão**, que trazem mais compras variadas.
- Transferência também tem as descrições mais longas em média, por causa do nome do banco no fim de cada Pix.
- Alguns casos são impossíveis de deduzir só pelo texto (ex.: `A B & B Comercio De Pr...`, razão social genérica). Eles vão aparecer na análise de erros.
---
 
## Como rodar
 
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```
 
Depois, abra `notebooks/01_exploracao.ipynb` e selecione o kernel `.venv`.
 
---
 
## Tecnologias
Python 3.11 · pandas · matplotlib · Jupyter · *(em breve)* scikit-learn, Hugging Face Transformers
 