# Brazilian Synthetic E-Commerce Public Dataset (by NerdDataLab)

Dataset de e-commerce brasileiro para a disciplina **Big Data e Fundamentos da Engenharia de Dados**.
Ele **continua** o dataset público do Olist (2016 a agosto de 2018) com pedidos **sintéticos**
de setembro de 2018 até novembro de 2026, em escala de milhões de pedidos.

**Você não gera nada.** Os arquivos já estão neste repositório, em `dados/`. O seu trabalho é
ler, ingerir, tratar e modelar esses dados com **Spark**, como faria um engenheiro de dados no dia a dia.

> ## Atenção: o que é real e o que é sintético
> | Período | Origem |
> |---|---|
> | 2016-09 a 2018-08 | **Real**: Brazilian E-Commerce Public Dataset by Olist (pasta `dados/olist/`) |
> | 2018-09 em diante | **Sintético**: pasta `dados/sintetico/`. Nenhum pedido, cliente ou vendedor deste período existe de verdade. |
>
> O gerador aprendeu as distribuições do dado real (estados, horários, preços, frete, prazos, notas)
> e aplicou tendências calibradas por dados públicos agregados do mercado (crescimento, sazonalidade, Pix, inflação).
> Serve para **ensino**. Não use para tirar conclusões sobre o mercado real.

## Créditos e licença

- Dados reais: **Olist**, publicados no Kaggle como *Brazilian E-Commerce Public Dataset by Olist*
  (<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>), licença **CC BY-NC-SA 4.0**
  (uso não comercial, com atribuição e mesma licença nas obras derivadas).
  Confira a licença vigente na página original.
- Este repositório é uma obra derivada, mantida pelo **NerdDataLab**, e **não é publicado nem endossado pela Olist**.
- Portanto, este repositório segue a mesma licença: **CC BY-NC-SA 4.0**, uso não comercial.

## Estrutura

```
dados/
  olist/                 CSVs reais do Olist (2016-2018), 1 arquivo por tabela
  sintetico/
    dimensoes/           produtos, vendedores e tradução de categorias (reais + novos)
    fatos/<tabela>/      histórico 2018-09 a 2026-10: 1 arquivo CSV por MÊS, para cada tabela
    stream/<tabela>/     últimos 30 dias (2026-10-29 a 2026-11-27): 1 arquivo CSV por DIA
gerador/gerar_dados.py   Código que gerou dados/sintetico/ (referência; você não precisa executar)
requirements.txt         dependências do gerador (numpy e pandas)
```

As tabelas de fatos seguem o esquema do Olist: `orders`, `customers`, `order_items`,
`order_payments` e `order_reviews`. Os nomes das colunas são os mesmos do dado real.

Tamanho: **641 arquivos CSV**, cerca de **1,3 GB** descompactados em `dados/sintetico/`
(o maior arquivo tem menos de 8 MB). O repositório inteiro baixa cerca de 500 MB (o Git comprime).

Pontos de atenção, de propósito, para você descobrir e tratar:

- O dado vem **sujo**, como na vida real. Não existe lista dos problemas: encontrá-los faz parte da atividade.
- O **esquema muda ao longo do tempo** (schema drift). Leia com cuidado antes de unir períodos.
- Alguns arquivos têm campos de texto com **quebra de linha dentro das aspas** (veja `order_reviews`).
- Pedidos podem chegar **depois** do mês em que aconteceram (dado atrasado).

## Como obter os dados

**No Google Colab** (recomendado: não usa o seu disco nem a sua instalação)
```python
!git clone --depth 1 https://github.com/wuldson-franco/Brazilian-Synthetic-E-Commerce-Public-Dataset-by-NerdDataLab repo
BASE = "/content/repo/dados"
```
`--depth 1` baixa só a versão atual, sem o histórico do Git.

**Local** (Git Bash, PowerShell, Linux ou macOS)
```
git clone --depth 1 https://github.com/wuldson-franco/Brazilian-Synthetic-E-Commerce-Public-Dataset-by-NerdDataLab
```
Se quiser baixar menos, pegue só o que vai usar (exemplo: sem a pasta `stream/`):
```
git clone --depth 1 --filter=blob:none --sparse https://github.com/wuldson-franco/Brazilian-Synthetic-E-Commerce-Public-Dataset-by-NerdDataLab
cd Brazilian-Synthetic-E-Commerce-Public-Dataset-by-NerdDataLab
git sparse-checkout set dados/olist dados/sintetico/dimensoes dados/sintetico/fatos
```

> No Windows, o repositório traz um `.gitattributes` que impede o Git de trocar o fim de linha dos CSVs.
> Não apague esse arquivo: ele mantém os CSVs idênticos aos originais.

## Como ler com Spark

Aponte para a **pasta da tabela**: o Spark lê todos os arquivos dela de uma vez.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("bigdata").getOrCreate()

BASE = "/content/repo/dados"          # ajuste ao seu caminho

# Histórico sintético: 1 pasta por tabela, vários arquivos mensais
orders = (spark.read
          .option("header", True)
          .csv(f"{BASE}/sintetico/fatos/orders/"))

print(orders.count(), len(orders.columns))
orders.printSchema()
```

Sem `inferSchema`, todas as colunas chegam como texto (`string`). Isso é deliberado na camada **Bronze**:
você decide os tipos na **Silver**.

`order_reviews` tem texto com quebra de linha entre aspas. Para ler sem quebrar registros:

```python
reviews = (spark.read
           .option("header", True)
           .option("multiLine", True)
           .option("quote", '"')
           .option("escape", '"')
           .csv(f"{BASE}/sintetico/fatos/order_reviews/"))
```

Na pasta `stream/` há um arquivo por dia. Ela serve para simular a chegada contínua de dados
(Structured Streaming com fonte de arquivos).

## Parâmetros usados na geração (referência)

O conteúdo de `dados/sintetico/` foi gerado com:

```
python gerador/gerar_dados.py --pedidos 2000000 --semente 42 --fim 2026-11-27
```

Com a mesma semente e os mesmos parâmetros, o gerador reproduz exatamente os mesmos arquivos
(testado em Linux e Windows, com pandas 2.3 e 3.0). Não é preciso fazer isso para a atividade.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `--pedidos` | 2000000 | Total de pedidos sintéticos |
| `--semente` | 42 | Semente aleatória |
| `--inicio` / `--fim` | 2018-09-01 / hoje | Período gerado |
| `--stream-dias` | 30 | Os últimos N dias saem em arquivos diários (pasta `stream/`) |
| `--sujeira` | 1.0 | Multiplicador dos problemas de qualidade injetados (0 desliga) |
| `--olist` | dados/olist | Pasta com os CSVs reais |

## Como o dado sintético foi calibrado

Cada número abaixo é ou **FONTE** (dado público agregado) ou **PREMISSA** (escolha do autor, para fins didáticos).
Os valores completos estão no topo do arquivo `gerador/gerar_dados.py`.

| Aspecto | Valor usado | Tipo |
|---|---|---|
| Pedidos por ano (crescimento) | 2023: +7%, 2024: +5,0%, 2025: +5,8%; 2020: +40% | FONTE (ABComm/ABIACOM) |
| Pedidos por ano, demais anos | 2019: +20%, 2021: +20%, 2022: +5%, 2026: +5% | PREMISSA (fontes divergem ou não há resultado fechado) |
| Black Friday | O fim de semana (sex a seg) pesa cerca de 4,5% do ano | FONTE (ABComm, 2024: 4,55%) |
| Peso do 4º trimestre | cerca de 33% do ano | FONTE qualitativa (mais de 30% em muitas lojas) |
| Pix | 0% até nov/2020; cerca de 30% em 2023, 40% em 2024, 38% em 2025 | FONTE (várias, com métricas diferentes) |
| Cartão, boleto e carteira digital | 2024: cartão 43%, boleto 8%, carteira 7% | FONTE + PREMISSA |
| Black Friday: meios de pagamento | crédito 63%, Pix 25,5%, boleto 7,4% | FONTE (2024) |
| Inflação (preços) | IPCA anual de 2019 a 2025 aplicado aos preços; 2026 em 4,5% | FONTE (IBGE) + PREMISSA para 2026 |

Fontes consultadas: ABComm/ABIACOM (via E-Commerce Brasil e Climba), PCMI e Let's Money (Pix),
Central do Varejo (Black Friday 2025), IBGE (IPCA). Os números do setor vêm de associações e imprensa
especializada: confira o relatório original antes de citá-los em trabalhos.

## Limitações conhecidas

- Os dados sintéticos **não têm correlações reais** além das que o gerador modela de propósito.
- 2 milhões de pedidos (cerca de 1,3 GB) **não é Big Data pelo padrão da indústria**. O objetivo é ver o comportamento
  das ferramentas (partições, formato colunar, custo de cada passada) que se mantém quando o volume cresce.
- O arquivo de geolocalização do Olist não está neste repositório.
- Em setembro de 2018 há uma **mudança de nível** no volume de pedidos (do real para o sintético). É um efeito do gerador.
