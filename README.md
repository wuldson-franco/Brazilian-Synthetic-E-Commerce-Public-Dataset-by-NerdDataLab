# Brazilian Synthetic E-Commerce Public Dataset (by NerdDataLab)

Dataset de e-commerce brasileiro para aulas de **Big Data e Fundamentos da Engenharia de Dados**.
Ele **continua** o dataset público do Olist (2016 a agosto de 2018) com pedidos
**sintéticos** de setembro de 2018 até hoje, em escala de milhões de pedidos.

> ## Atenção: o que é real e o que é sintético
> | Período | Origem |
> |---|---|
> | 2016-09 a 2018-08 | **Real**: Brazilian E-Commerce Public Dataset by Olist (pasta `dados/olist/`) |
> | 2018-09 em diante | **Sintético**: gerado por `gerador/gerar_dados.py`. Nenhum pedido, cliente ou vendedor deste período existe de verdade. |
>
> O gerador aprende as distribuições do dado real (estados, horários, preços, frete, prazos, notas)
> e aplica tendências calibradas por dados públicos agregados do mercado (crescimento, sazonalidade, Pix, inflação).
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
dados/olist/            CSVs reais do Olist (2016-2018). Não altere: a camada Bronze parte daqui.
gerador/gerar_dados.py  Gerador dos dados sintéticos (2018-09 em diante)
requirements.txt        numpy e pandas
saida/                  (criada ao gerar; não vai para o Git)
```

## Como gerar os dados

Requer Python 3.10+, `numpy` e `pandas` (já instalados no Google Colab).

**No Google Colab**
```python
!git clone https://github.com/wuldson-franco/brazilian-sintetic-e-commerce-public-dataset-by-nerddatalab
%cd brazilian-sintetic-e-commerce-public-dataset-by-nerddatalab
!python gerador/gerar_dados.py --pedidos 2000000 --saida /content/saida
```

**Local (Windows PowerShell, Linux ou macOS)**
```
pip install -r requirements.txt
python gerador/gerar_dados.py --pedidos 2000000 --saida saida
```

Com a mesma `--semente` (padrão 42), **todos recebem exatamente os mesmos arquivos**.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `--pedidos` | 2000000 | Total de pedidos sintéticos |
| `--semente` | 42 | Semente aleatória (mesma semente = mesmos dados) |
| `--inicio` / `--fim` | 2018-09-01 / hoje | Período gerado |
| `--stream-dias` | 30 | Os últimos N dias saem em arquivos diários (pasta `stream/`) |
| `--sujeira` | 1.0 | Multiplicador dos problemas de qualidade injetados (0 desliga) |
| `--olist` | dados/olist | Pasta com os CSVs reais |

## O que é gerado

```
saida/
  dimensoes/                 produtos, vendedores e tradução de categorias (reais + novos)
  fatos/<tabela>/            histórico: 1 arquivo CSV por MÊS, para cada tabela
  stream/<tabela>/           últimos N dias: 1 arquivo CSV por DIA (para simular streaming)
  _gabarito/                 reservado ao professor (veja o aviso abaixo)
```

As tabelas de fatos seguem o esquema do Olist: `orders`, `customers`, `order_items`,
`order_payments` e `order_reviews`. Os nomes das colunas são os mesmos do dado real.

Tamanho medido (escala padrão de 2 milhões de pedidos): cerca de **1,2 GB** em CSV e **644 arquivos**,
gerados em cerca de 50 segundos numa máquina de 2 núcleos.

> **Sobre `_gabarito/`:** essa pasta lista os problemas injetados nos dados. Ela existe para a correção
> das atividades. **Não abra antes de terminar a sua análise**, senão você perde o exercício de descobrir os problemas.

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
- 2 milhões de pedidos (cerca de 1,2 GB) **não é Big Data pelo padrão da indústria**. O objetivo é ver o comportamento
  das ferramentas (partições, formato colunar, custo de cada passada) que se mantém quando o volume cresce.
- O arquivo de geolocalização do Olist não está neste repositório.
- Em setembro de 2018 há uma **mudança de nível** no volume de pedidos (do real para o sintético). É um efeito do gerador.