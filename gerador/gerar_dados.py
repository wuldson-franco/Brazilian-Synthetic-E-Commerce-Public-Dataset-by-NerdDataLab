#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de dados SINTETICOS de e-commerce, no esquema do Olist (Kaggle).

Continua o dataset real do Olist (2016 a ago/2018) de 2018-09 ate hoje.
Tudo que vem depois de agosto de 2018 e SINTETICO: nenhum pedido abaixo e real.

Como funciona
  1. APRENDE as distribuicoes dos CSVs reais do Olist (estados, horarios, itens por
     pedido, precos, frete, prazos, notas de avaliacao...). Nada e copiado pedido a
     pedido, exceto os textos de comentario das avaliacoes (amostrados do Olist).
  2. GERA pedidos novos, mes a mes, com macro-tendencias calibradas por dados
     publicos agregados (crescimento do e-commerce, sazonalidade, Pix, IPCA).
     Veja o dicionario PREMISSAS abaixo: cada numero diz se tem fonte ou e premissa.
  3. INJETA sujeira controlada (duplicatas, nulos, datas inconsistentes, chegada
     tardia...) e grava um GABARITO do que foi injetado.

Reprodutivel: a mesma --semente gera exatamente os mesmos arquivos.

Uso
  python gerador/gerar_dados.py --pedidos 2000000
  python gerador/gerar_dados.py --pedidos 100000 --saida saida_teste   (teste rapido)
"""
import argparse
import json
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------
# PREMISSAS (macro-ancoras). "FONTE" = dado publico agregado; "PREMISSA" = nossa escolha.
# --------------------------------------------------------------------------------
# Crescimento anual do NUMERO DE PEDIDOS (multiplicador sobre o ano anterior).
CRESCIMENTO_ANUAL = {
    2019: 1.20,   # PREMISSA
    2020: 1.40,   # FONTE (ABComm: crescimento de mais de 40% entre 2019 e 2020; vale p/ faturamento)
    2021: 1.20,   # PREMISSA (as fontes divergem bastante para 2021)
    2022: 1.05,   # PREMISSA (desaceleracao pos-pandemia)
    2023: 1.07,   # FONTE derivada (faturamento +9,5% com ticket +2%)
    2024: 1.050,  # FONTE (ABComm: 395 mi -> 414,9 mi pedidos)
    2025: 1.058,  # FONTE (ABComm/ABIACOM: 414,9 mi -> 438,9 mi pedidos)
    2026: 1.05,   # PREMISSA (sem resultado fechado)
    2027: 1.05,   # apenas para extrapolar a curva
}
# Sazonalidade mensal (PREMISSA coerente com a fonte: o 4o trimestre pesa >30% do ano).
SAZONALIDADE_MES = {1: .90, 2: .88, 3: .97, 4: .95, 5: 1.05, 6: .95,
                    7: .92, 8: .93, 9: .95, 10: 1.00, 11: 1.45, 12: 1.30}
# Dia da semana (FONTE qualitativa: pico entre domingo e segunda). Seg=0 ... Dom=6.
SAZONALIDADE_DOW = {0: 1.10, 1: 1.05, 2: 1.00, 3: .98, 4: .95, 5: .85, 6: 1.07}
# FONTE (ABComm, 2024): o fim de semana da Black Friday = 4,55% da receita do ano.
FATIA_BLACK_FRIDAY = 0.045
# IPCA anual (IBGE). 2026 e PREMISSA (so ha 6 meses fechados).
IPCA_ANUAL = {2018: .0375, 2019: .0431, 2020: .0452, 2021: .1006, 2022: .0579,
              2023: .0462, 2024: .0483, 2025: .0426, 2026: .045}
# Meios de pagamento. Pix FONTE ~30% (2023), 40% (2024), 38% (2025). Demais: PREMISSA
# ajustada as fontes (cartao 44%, boleto 8%, carteira digital 7% em 2024).
TIPOS = ["credit_card", "boleto", "voucher", "debit_card", "pix", "digital_wallet"]
LANCAMENTO_PIX = date(2020, 11, 16)
ANCORAS_PAGAMENTO = [  # (data, [credit_card, boleto, voucher, debit_card, pix, digital_wallet])
    (date(2020, 11, 16), None),  # None = mix real do Olist
    (date(2021, 12, 31), [.60, .14, .03, .02, .15, .06]),
    (date(2023, 6, 30),  [.50, .10, .01, .01, .30, .08]),
    (date(2024, 6, 30),  [.43, .08, .01, .01, .40, .07]),
    (date(2025, 6, 30),  [.46, .07, .01, .01, .38, .07]),
    (date(2026, 1, 1),   [.44, .07, .01, .01, .40, .07]),
]
MIX_BLACK_FRIDAY = [.63, .074, .006, .005, .255, .03]  # FONTE: BF 2024 (credito 63%, Pix 25,5%, boleto 7,4%)
# Chance de sellers "ruins" (atrasam mais): PREMISSA, serve de sinal para ML.
FRACAO_SELLER_RUIM = 0.10
# Troca de esquema (schema drift)
INICIO_DELIVERY_MODE = date(2024, 1, 1)  # coluna nova em orders a partir daqui


# --------------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------------
def hexids(rng, n):
    """IDs de 32 caracteres hexadecimais, como no Olist."""
    if n == 0:
        return np.array([], dtype=object)
    h = rng.bytes(16 * n).hex()
    return np.array([h[i:i + 32] for i in range(0, 32 * n, 32)], dtype=object)


def amostra(arr, n, rng):
    return arr[rng.integers(0, len(arr), size=n)]


def black_friday(ano):
    d = date(ano, 11, 1)
    quinta1 = d + timedelta((3 - d.weekday()) % 7)
    return quinta1 + timedelta(days=22)  # sexta depois da 4a quinta de novembro


def fmt(serie_dt):
    return pd.Series(serie_dt).dt.strftime("%Y-%m-%d %H:%M:%S")


def frequencias(serie):
    """Frequencias relativas com desempate EXPLICITO (por nome). O desempate padrao do
    value_counts muda entre versoes do pandas e quebraria a reprodutibilidade."""
    c = serie.value_counts()
    nomes = np.array([str(x) for x in c.index])
    ordem = np.lexsort((nomes, -c.values))
    return nomes[ordem], c.values[ordem] / c.values.sum()


# --------------------------------------------------------------------------------
# 1) APRENDER com o Olist real
# --------------------------------------------------------------------------------
def aprender(pasta):
    L = lambda n: pd.read_csv(os.path.join(pasta, n))
    cli, ped, it = L("olist_customers_dataset.csv"), L("olist_orders_dataset.csv"), L("olist_order_items_dataset.csv")
    pag, rev = L("olist_order_payments_dataset.csv"), L("olist_order_reviews_dataset.csv")
    pro, ven, tra = L("olist_products_dataset.csv"), L("olist_sellers_dataset.csv"), L("product_category_name_translation.csv")

    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"]:
        ped[c] = pd.to_datetime(ped[c])
    it["shipping_limit_date"] = pd.to_datetime(it["shipping_limit_date"])
    rev["review_creation_date"] = pd.to_datetime(rev["review_creation_date"])
    rev["review_answer_timestamp"] = pd.to_datetime(rev["review_answer_timestamp"])

    ped = ped.merge(cli[["customer_id", "customer_state"]], on="customer_id")
    prim = (it.sort_values("order_item_id").drop_duplicates("order_id")[["order_id", "seller_id"]]
            .merge(ven[["seller_id", "seller_state"]], on="seller_id"))
    ped = ped.merge(prim[["order_id", "seller_state"]], on="order_id", how="left")
    ped["same"] = ped.customer_state == ped.seller_state

    A = {}
    A["cli_loc"] = cli[["customer_state", "customer_city", "customer_zip_code_prefix"]].reset_index(drop=True)
    A["ven_loc"] = ven[["seller_state", "seller_city", "seller_zip_code_prefix"]].reset_index(drop=True)
    A["hora_p"] = ped.order_purchase_timestamp.dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0).values
    vc = it.groupby("order_id").size().value_counts(normalize=True).sort_index()
    A["n_itens_v"], A["n_itens_p"] = vc.index.values, vc.values

    # pares (produto, seller) com popularidade real
    g = it.groupby(["product_id", "seller_id"]).agg(n=("order_id", "size"), preco=("price", "median")).reset_index()
    pro_i = pro.reset_index(drop=True)
    pro_idx = {p: i for i, p in enumerate(pro_i.product_id)}
    ven_idx = {s: i for i, s in enumerate(ven.seller_id)}
    A["par_prod"] = g.product_id.map(pro_idx).values
    A["par_sell"] = g.seller_id.map(ven_idx).values
    A["par_peso"] = g.n.values.astype(float)
    A["par_preco"] = g.preco.values
    A["produtos"], A["vendedores"], A["traducao"] = pro_i, ven.reset_index(drop=True), tra
    A["peso_mediano"] = float(pro_i.product_weight_g.median())

    # frete ~ lognormal por (mesmo estado?, faixa de peso)
    itf = (it.merge(ped[["order_id", "customer_state"]], on="order_id")
             .merge(ven[["seller_id", "seller_state"]], on="seller_id")
             .merge(pro[["product_id", "product_weight_g"]], on="product_id", how="left"))
    itf["same"] = (itf.customer_state == itf.seller_state).astype(int)
    w = itf.product_weight_g.fillna(A["peso_mediano"])
    A["peso_bordas"] = np.unique(np.quantile(w, np.linspace(.1, .9, 9)))
    itf["wb"] = np.searchsorted(A["peso_bordas"], w)
    itf["lf"] = np.log1p(itf.freight_value)
    tab = itf.groupby(["same", "wb"]).lf.agg(["mean", "std"]).reset_index()
    nb = len(A["peso_bordas"]) + 1
    mu = np.full((2, nb), itf.lf.mean()); sd = np.full((2, nb), itf.lf.std())
    for _, r in tab.iterrows():
        mu[int(r["same"]), int(r["wb"])] = r["mean"]
        if not np.isnan(r["std"]):
            sd[int(r["same"]), int(r["wb"])] = r["std"]
    A["frete_mu"], A["frete_sd"] = mu, sd

    # precos por categoria (para produtos novos)
    cat = pro_i.product_category_name.fillna("__sem__")
    pc = it.merge(pro[["product_id", "product_category_name"]], on="product_id")
    pc["lp"] = np.log(pc.price.clip(lower=1))
    pc["cat"] = pc.product_category_name.fillna("__sem__")
    st = pc.groupby("cat").lp.agg(["mean", "std"]).fillna({"std": 0.5})
    A["cat_nomes"], A["cat_prob"] = frequencias(cat)
    A["cat_preco"] = {c: (st.loc[c, "mean"], max(st.loc[c, "std"], 0.2)) if c in st.index else (4.2, 0.8)
                      for c in A["cat_nomes"]}
    A["cat_modelos"] = {c: np.where(cat.values == c)[0] for c in A["cat_nomes"]}

    # linha do tempo do pedido
    ent = ped[(ped.order_status == "delivered") & ped.order_delivered_customer_date.notna()
              & ped.order_delivered_carrier_date.notna() & ped.order_approved_at.notna()].copy()
    sec = lambda a, b: (a - b).dt.total_seconds()
    A["aprov_s"] = sec(ped.order_approved_at, ped.order_purchase_timestamp).dropna().clip(lower=0).values
    A["manus_s"] = sec(ent.order_delivered_carrier_date, ent.order_approved_at).clip(lower=0).values
    for k, flag in (("same", True), ("diff", False)):
        e = ent[ent.same == flag]
        A[f"trans_{k}"] = sec(e.order_delivered_customer_date, e.order_delivered_carrier_date).clip(lower=3600).values
        e2 = ped[ped.same == flag]
        A[f"est_{k}"] = (e2.order_estimated_delivery_date.dt.normalize() - e2.order_purchase_timestamp.dt.normalize()).dt.days.clip(lower=2).values
    itm = it.merge(ped[["order_id", "order_approved_at"]], on="order_id")
    A["limite_s"] = sec(itm.shipping_limit_date, itm.order_approved_at).dropna().clip(lower=0).values
    A["p_same"] = float(ped.same.mean())
    ee = ped[ped.order_delivered_customer_date.notna()]
    A["atraso_real"] = float((ee.order_delivered_customer_date.dt.normalize() > ee.order_estimated_delivery_date).mean())
    A["status_v"], A["status_p"] = frequencias(ped.order_status)

    # avaliacoes: nota depende de atraso
    r = rev.merge(ped[["order_id", "order_status", "order_delivered_customer_date", "order_estimated_delivery_date"]], on="order_id")
    ent_ok = r.order_delivered_customer_date.notna()
    atras = ent_ok & (r.order_delivered_customer_date.dt.normalize() > r.order_estimated_delivery_date)
    dist = lambda m: r[m].review_score.value_counts(normalize=True).reindex(range(1, 6), fill_value=0).values
    A["nota_atraso"], A["nota_ok"], A["nota_sem_entrega"] = dist(atras), dist(ent_ok & ~atras), dist(~ent_ok)
    A["pool_coment"] = {s: rev[rev.review_score == s][["review_comment_title", "review_comment_message"]].reset_index(drop=True)
                        for s in range(1, 6)}
    A["resp_s"] = sec(rev.review_answer_timestamp, rev.review_creation_date).dropna().clip(lower=0).values

    # pagamentos
    p1 = pag[pag.payment_sequential == 1]
    mix = p1.payment_type.value_counts(normalize=True).reindex(TIPOS, fill_value=0)
    A["mix_olist"] = (mix / mix.sum()).values
    A["parcelas"] = pag.loc[pag.payment_type == "credit_card", "payment_installments"].clip(lower=1).values
    A["p_multi"] = float((pag.groupby("order_id").size() > 1).mean())
    return A


def calibrar_estimativa(A, rng):
    """Acha k tal que, sem efeitos plantados, o atraso simulado = atraso real do Olist.
    (Sortear prazos de forma independente infla a variancia; k reajusta a estimativa.)"""
    n = 200_000
    same = rng.random(n) < A["p_same"]
    t0 = rng.choice(24, n, p=A["hora_p"] / A["hora_p"].sum()) * 3600 + rng.integers(0, 3600, n)
    total = amostra(A["aprov_s"], n, rng) + amostra(A["manus_s"], n, rng) + np.where(
        same, amostra(A["trans_same"], n, rng), amostra(A["trans_diff"], n, rng))
    est = np.where(same, amostra(A["est_same"], n, rng), amostra(A["est_diff"], n, rng)).astype(float)
    dia_entrega = np.floor((t0 + total) / 86400)
    lo, hi = 0.8, 2.0
    for _ in range(30):
        k = (lo + hi) / 2
        taxa = (dia_entrega > np.rint(est * k)).mean()
        lo, hi = (k, hi) if taxa > A["atraso_real"] else (lo, k)
    return (lo + hi) / 2


# --------------------------------------------------------------------------------
# 2) CALENDARIO: quantos pedidos por dia
# --------------------------------------------------------------------------------
def indice_nivel(datas):
    anos = sorted(CRESCIMENTO_ANUAL)
    nivel, ords, vals = 1.0, [pd.Timestamp(2017, 7, 1).toordinal()], [np.log(1 / CRESCIMENTO_ANUAL[2019])]
    ords.append(pd.Timestamp(2018, 7, 1).toordinal()); vals.append(0.0)
    for a in anos:
        nivel *= CRESCIMENTO_ANUAL[a]
        ords.append(pd.Timestamp(a, 7, 1).toordinal()); vals.append(np.log(nivel))
    x = np.array([d.toordinal() for d in datas])
    return np.exp(np.interp(x, ords, vals))


def calendario(inicio, fim):
    """Peso de cada dia = tendencia * mes * dia da semana * pico da Black Friday."""
    dias = pd.date_range("2018-01-01", "2026-12-31")
    w = (indice_nivel(dias) * dias.month.map(SAZONALIDADE_MES).values.astype(float)
         * dias.dayofweek.map(SAZONALIDADE_DOW).values.astype(float))
    w = pd.Series(w, index=dias)
    bf_flag = pd.Series(False, index=dias)
    for ano in range(2018, 2027):
        bf = pd.Timestamp(black_friday(ano))
        jan = pd.date_range(f"{ano}-01-01", f"{ano}-12-31")
        b = pd.date_range(bf, bf + pd.Timedelta(days=3))  # sex a seg (Cyber Monday)
        soma_b, soma_o = w.loc[b].sum(), w.loc[jan].sum() - w.loc[b].sum()
        k = FATIA_BLACK_FRIDAY * soma_o / ((1 - FATIA_BLACK_FRIDAY) * soma_b)
        w.loc[b] *= k
        bf_flag.loc[b] = True
    w, bf_flag = w.loc[inicio:fim], bf_flag.loc[inicio:fim]
    return w, bf_flag


def inflacao(datas):
    """Fator de preco acumulado desde 2018-09-01, pelo IPCA anual (juros compostos mensais)."""
    base = pd.Timestamp(2018, 9, 1)
    meses = (datas.year - base.year) * 12 + (datas.month - base.month)
    out = np.empty(len(datas))
    cache = {}
    for i, (a, m) in enumerate(zip(datas.year, meses)):
        if m not in cache:
            f = 1.0
            for k in range(m):
                ano = (base.month - 1 + k) // 12 + base.year
                f *= (1 + IPCA_ANUAL.get(ano, .045)) ** (1 / 12)
            cache[m] = f
        out[i] = cache[m]
    return out


def mix_pagamento(dia, mix_olist):
    d = dia
    if d < LANCAMENTO_PIX:
        return np.array(mix_olist)
    pontos = [(a, np.array(mix_olist) if m is None else np.array(m)) for a, m in ANCORAS_PAGAMENTO]
    if d >= pontos[-1][0]:
        return pontos[-1][1]
    for (d0, m0), (d1, m1) in zip(pontos[:-1], pontos[1:]):
        if d0 <= d < d1:
            t = (d - d0).days / max((d1 - d0).days, 1)
            return (1 - t) * m0 + t * m1


# --------------------------------------------------------------------------------
# 3) GERAR
# --------------------------------------------------------------------------------
def montar_catalogo(A, rng, n_prod_novos, n_sell_novos, inicio, fim):
    """Catalogo = produtos/sellers reais do Olist + novos sinteticos (com data de entrada)."""
    pro, ven = A["produtos"].copy(), A["vendedores"].copy()
    n_old_p, n_old_s = len(pro), len(ven)
    span = (fim - inicio).days

    # sellers novos
    loc = A["ven_loc"].iloc[rng.integers(0, len(A["ven_loc"]), n_sell_novos)].reset_index(drop=True)
    sel_novos = pd.DataFrame({"seller_id": hexids(rng, n_sell_novos), "seller_zip_code_prefix": loc.seller_zip_code_prefix,
                              "seller_city": loc.seller_city, "seller_state": loc.seller_state})
    ven = pd.concat([ven, sel_novos], ignore_index=True)
    ativ_s = np.concatenate([np.zeros(n_old_s), rng.integers(0, span, n_sell_novos)])
    ruim = rng.random(len(ven)) < FRACAO_SELLER_RUIM
    pop_s = 1.0 / np.arange(1, len(ven) + 1) ** 0.9
    pop_s = pop_s[rng.permutation(len(ven))]

    # produtos novos
    cats = rng.choice(A["cat_nomes"], size=n_prod_novos, p=A["cat_prob"])
    modelos = np.array([rng.choice(A["cat_modelos"][c]) for c in cats])
    base = pro.iloc[modelos].reset_index(drop=True).copy()
    base["product_id"] = hexids(rng, n_prod_novos)
    for c in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        base[c] = (base[c] * rng.lognormal(0, .15, n_prod_novos)).round()
    preco_novo = np.array([np.exp(rng.normal(*A["cat_preco"][c])) for c in cats]).clip(5, 6000)
    # cada produto novo e vendido por um seller (sorteado por popularidade)
    sell_p = rng.choice(len(ven), size=n_prod_novos, p=pop_s / pop_s.sum())
    ativ_p = np.maximum(rng.integers(0, span, n_prod_novos), ativ_s[sell_p])
    pro = pd.concat([pro, base[pro.columns]], ignore_index=True)
    pop_p = 1.0 / np.arange(1, n_prod_novos + 1) ** 1.0
    pop_p = pop_p[rng.permutation(n_prod_novos)]

    cat_ = {
        "old_prod": A["par_prod"], "old_sell": A["par_sell"], "old_w": A["par_peso"], "old_preco": A["par_preco"],
        "new_prod": n_old_p + np.arange(n_prod_novos), "new_sell": sell_p, "new_w": pop_p,
        "new_preco": preco_novo, "new_ativ": ativ_p,
        "pro": pro, "ven": ven, "ruim": ruim,
        "peso": pro.product_weight_g.fillna(A["peso_mediano"]).values,
        "sell_state": ven.seller_state.values,
    }
    return cat_


def gerar(args):
    t0 = time.time()
    rng_mestre = np.random.SeedSequence(args.semente)
    rng = np.random.default_rng(rng_mestre.spawn(1)[0])
    inicio, fim = pd.Timestamp(args.inicio), pd.Timestamp(args.fim)
    print(f"[1/4] Aprendendo com o Olist real em {args.olist} ...")
    A = aprender(args.olist)
    A["k_est"] = calibrar_estimativa(A, rng)
    print(f"      atraso real do Olist = {A['atraso_real']:.3f}; fator de calibracao do prazo estimado k = {A['k_est']:.3f}")

    escala = args.pedidos / 2_000_000
    cat = montar_catalogo(A, rng, max(int(20000 * escala), 2000), max(int(3000 * escala), 300), inicio, fim)

    print("[2/4] Calendario e volume por dia ...")
    pesos, bf = calendario(inicio, fim)
    cont = rng.multinomial(args.pedidos, pesos.values / pesos.values.sum())
    dias = pesos.index
    cont = pd.Series(cont, index=dias)

    saida = args.saida
    for sub in ["dimensoes", "_gabarito"]:
        os.makedirs(os.path.join(saida, sub), exist_ok=True)
    cat["pro"].to_csv(os.path.join(saida, "dimensoes", "olist_products_dataset.csv"), index=False)
    cat["ven"].drop(columns=[]).to_csv(os.path.join(saida, "dimensoes", "olist_sellers_dataset.csv"), index=False)
    A["traducao"].to_csv(os.path.join(saida, "dimensoes", "product_category_name_translation.csv"), index=False)

    fim_ts = fim + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    stream_ini = (fim - pd.Timedelta(days=args.stream_dias - 1)).normalize() if args.stream_dias > 0 else fim + pd.Timedelta(days=9)

    pool_uid, pool_loc = [], []
    gab, resumo_meses = [], []
    carry = {}  # linhas que chegam atrasadas no mes seguinte
    meses = pd.period_range(inicio, fim, freq="M")
    print(f"[3/4] Gerando {args.pedidos:,} pedidos em {len(meses)} meses ...")

    for mi, per in enumerate(meses):
        dias_m = dias[(dias >= per.start_time) & (dias <= per.end_time)]
        n_por_dia = cont.loc[dias_m].values
        n = int(n_por_dia.sum())
        if n == 0:
            continue
        r = np.random.default_rng(rng_mestre.spawn(1)[0])

        # ---------- pedidos: data/hora de compra ----------
        dia_idx = np.repeat(np.arange(len(dias_m)), n_por_dia)
        dia_ts = dias_m.values[dia_idx].astype("datetime64[s]")
        hora = r.choice(24, size=n, p=A["hora_p"] / A["hora_p"].sum())
        compra = dia_ts + (hora * 3600 + r.integers(0, 3600, n)).astype("timedelta64[s]")
        compra_dt = pd.DatetimeIndex(compra)
        infl = inflacao(compra_dt)
        e_bf = bf.loc[dias_m].values[dia_idx]
        bf_ano = np.array([pd.Timestamp(black_friday(a)) for a in dias_m.year.unique()])
        dentro_bf = np.zeros(n, bool)
        for b in bf_ano:  # janela de maior atraso logistico
            dentro_bf |= (compra_dt >= b - pd.Timedelta(days=5)) & (compra_dt <= b + pd.Timedelta(days=12))

        # ---------- clientes ----------
        rep = (r.random(n) < 0.04) & (len(pool_uid) > 0)
        novos = ~rep
        uid = np.empty(n, dtype=object)
        loc_idx = np.empty(n, dtype=int)
        k_rep = int(rep.sum())
        if k_rep:
            ix = r.integers(0, len(pool_uid), k_rep)
            uid[rep] = np.array(pool_uid, dtype=object)[ix]
            loc_idx[rep] = np.array(pool_loc)[ix]
        k_new = int(novos.sum())
        uid[novos] = hexids(r, k_new)
        loc_idx[novos] = r.integers(0, len(A["cli_loc"]), k_new)
        pool_uid.extend(uid[novos].tolist()); pool_loc.extend(loc_idx[novos].tolist())
        L = A["cli_loc"].iloc[loc_idx].reset_index(drop=True)
        order_id, customer_id = hexids(r, n), hexids(r, n)
        clientes = pd.DataFrame({"customer_id": customer_id, "customer_unique_id": uid,
                                 "customer_zip_code_prefix": L.customer_zip_code_prefix,
                                 "customer_city": L.customer_city, "customer_state": L.customer_state})

        # ---------- itens ----------
        k = r.choice(A["n_itens_v"], size=n, p=A["n_itens_p"] / A["n_itens_p"].sum())
        m = int(k.sum())
        ped_do_item = np.repeat(np.arange(n), k)
        seq = np.arange(m) - np.repeat(np.cumsum(k) - k, k) + 1
        fracao_ano = ((compra_dt - inicio).days.values / max((fim - inicio).days, 1))
        p_novo = np.clip(0.5 * fracao_ano, 0, 0.5)[ped_do_item]
        usa_novo = r.random(m) < p_novo
        fim_mes_dias = (per.end_time - inicio).days
        elegivel = cat["new_ativ"] <= fim_mes_dias
        w_old = cat["old_w"] / cat["old_w"].sum()
        w_new = np.where(elegivel, cat["new_w"], 0.0)
        w_new = w_new / w_new.sum() if w_new.sum() > 0 else w_new
        sel = np.empty(m, dtype=int)
        io, inw = ~usa_novo, usa_novo
        sel_old = np.searchsorted(np.cumsum(w_old), r.random(int(io.sum())) * w_old.sum()).clip(0, len(w_old) - 1)
        prod_idx = np.empty(m, dtype=int); sell_idx = np.empty(m, dtype=int); preco_base = np.empty(m)
        prod_idx[io], sell_idx[io], preco_base[io] = cat["old_prod"][sel_old], cat["old_sell"][sel_old], cat["old_preco"][sel_old]
        if inw.any():
            sel_new = np.searchsorted(np.cumsum(w_new), r.random(int(inw.sum())) * w_new.sum()).clip(0, len(w_new) - 1)
            prod_idx[inw], sell_idx[inw], preco_base[inw] = cat["new_prod"][sel_new], cat["new_sell"][sel_new], cat["new_preco"][sel_new]
        infl_i = infl[ped_do_item]
        preco = (preco_base * infl_i * r.lognormal(0, .05, m)).round(2)
        cli_state = clientes.customer_state.values[ped_do_item]
        same_i = (cat["sell_state"][sell_idx] == cli_state).astype(int)
        wb = np.searchsorted(A["peso_bordas"], cat["peso"][prod_idx])
        frete = (np.expm1(r.normal(A["frete_mu"][same_i, wb], A["frete_sd"][same_i, wb])).clip(0) * infl_i).round(2)

        # ---------- linha do tempo do pedido ----------
        primeiro = np.cumsum(k) - k
        seller_o = sell_idx[primeiro]
        same_o = same_i[primeiro].astype(bool)
        ruim = cat["ruim"][seller_o]
        usa_dm = dia_ts.astype("datetime64[D]") >= np.datetime64(INICIO_DELIVERY_MODE.isoformat())
        modo = np.where(usa_dm, r.choice(["standard", "express", "pickup"], size=n, p=[.80, .15, .05]), None)
        aprov = compra + amostra(A["aprov_s"], n, r).astype("timedelta64[s]")
        manus = amostra(A["manus_s"], n, r) * (1 + 2.0 * ruim)
        carrier = aprov + manus.astype("timedelta64[s]")
        transito = np.where(same_o, amostra(A["trans_same"], n, r), amostra(A["trans_diff"], n, r)).astype(float)
        transito = transito * np.where(dentro_bf, 1.35, 1.0) * (1 + 0.30 * ruim) * np.where(modo == "express", 0.65, 1.0)
        entrega = carrier + transito.astype("timedelta64[s]")
        est_dias = np.rint(np.where(same_o, amostra(A["est_same"], n, r), amostra(A["est_diff"], n, r)) * A["k_est"]).astype(int)
        estimada = compra.astype("datetime64[D]").astype("datetime64[s]") + (est_dias * 86400).astype("timedelta64[s]")

        status = r.choice(A["status_v"], size=n, p=A["status_p"] / A["status_p"].sum())
        NaT = np.datetime64("NaT", "s")
        aprov_o, carr_o, ent_o = aprov.copy(), carrier.copy(), entrega.copy()
        ent_o[status != "delivered"] = NaT
        carr_o[np.isin(status, ["created", "approved", "invoiced", "processing", "unavailable"])] = NaT
        carr_o[(status == "canceled") & (r.random(n) < .8)] = NaT
        aprov_o[status == "created"] = NaT
        aprov_o[(status == "canceled") & (r.random(n) < .3)] = NaT
        # corte pela data final: o que ainda nao aconteceu fica vazio e o status acompanha
        fim_np = np.datetime64(fim_ts, "s")
        futuro_ent = (ent_o > fim_np)
        ent_o[futuro_ent] = NaT; status[futuro_ent] = "shipped"
        futuro_car = (carr_o > fim_np)
        carr_o[futuro_car] = NaT; ent_o[futuro_car] = NaT; status[futuro_car & (status == "shipped")] = "processing"
        futuro_ap = (aprov_o > fim_np)
        aprov_o[futuro_ap] = NaT; carr_o[futuro_ap] = NaT; ent_o[futuro_ap] = NaT; status[futuro_ap] = "created"

        ped_df = pd.DataFrame({
            "order_id": order_id, "customer_id": customer_id, "order_status": status,
            "order_purchase_timestamp": fmt(compra), "order_approved_at": fmt(aprov_o),
            "order_delivered_carrier_date": fmt(carr_o), "order_delivered_customer_date": fmt(ent_o),
            "order_estimated_delivery_date": fmt(estimada)})
        if usa_dm.any():
            ped_df["delivery_mode"] = modo

        lim = aprov[ped_do_item] + amostra(A["limite_s"], m, r).astype("timedelta64[s]")
        itens = pd.DataFrame({"order_id": order_id[ped_do_item], "order_item_id": seq,
                              "product_id": cat["pro"].product_id.values[prod_idx],
                              "seller_id": cat["ven"].seller_id.values[sell_idx],
                              "shipping_limit_date": fmt(lim), "price": preco, "freight_value": frete})

        # ---------- pagamentos ----------
        total = np.add.reduceat(preco + frete, primeiro)
        dia_d = [d.date() for d in dias_m]
        mixes = np.array([mix_pagamento(d, A["mix_olist"]) for d in dia_d])
        mixes_bf = mixes.copy()
        for i, d in enumerate(dia_d):
            if bf.loc[dias_m[i]] and d >= LANCAMENTO_PIX:
                mixes_bf[i] = MIX_BLACK_FRIDAY
        mx = mixes_bf[dia_idx]
        tipo_i = (r.random(n)[:, None] > np.cumsum(mx, axis=1)).sum(axis=1).clip(0, len(TIPOS) - 1)
        tipo = np.array(TIPOS, dtype=object)[tipo_i]
        parc = np.where(tipo == "credit_card", amostra(A["parcelas"], n, r), 1)
        multi = r.random(n) < A["p_multi"]
        v_voucher = (total * r.uniform(.1, .6, n)).round(2)
        pg1 = pd.DataFrame({"order_id": order_id, "payment_sequential": 1,
                            "payment_type": np.where(multi, "voucher", tipo),
                            "payment_installments": np.where(multi, 1, parc),
                            "payment_value": np.where(multi, v_voucher, total.round(2))})
        pg2 = pd.DataFrame({"order_id": order_id[multi], "payment_sequential": 2,
                            "payment_type": np.where(tipo[multi] == "voucher", "credit_card", tipo[multi]),
                            "payment_installments": parc[multi],
                            "payment_value": (total[multi] - v_voucher[multi]).round(2)})
        pagamentos = pd.concat([pg1, pg2], ignore_index=True)

        # ---------- avaliacoes ----------
        entregue = ~pd.isna(ent_o)
        atraso = entregue & (ent_o.astype("datetime64[D]") > estimada.astype("datetime64[D]"))
        p_aval = np.where(status == "delivered", .995, np.where(status == "canceled", .6, .5))
        tem = r.random(n) < p_aval
        base_dt = np.where(entregue, ent_o, estimada)
        cria = base_dt.astype("datetime64[D]").astype("datetime64[s]")
        resp = cria + amostra(A["resp_s"], n, r).astype("timedelta64[s]")
        tem &= resp <= fim_np
        extra = tem & (r.random(n) < .005)
        idx_r = np.concatenate([np.where(tem)[0], np.where(extra)[0]])
        nr = len(idx_r)
        P = np.where(atraso, 0, np.where(entregue, 1, 2))[idx_r]
        notas = np.empty(nr, dtype=int)
        for g, dist in ((0, A["nota_atraso"]), (1, A["nota_ok"]), (2, A["nota_sem_entrega"])):
            mk = P == g
            if mk.any():
                notas[mk] = r.choice(np.arange(1, 6), size=int(mk.sum()), p=dist / dist.sum())
        titulos = np.empty(nr, dtype=object); msgs = np.empty(nr, dtype=object)
        for s in range(1, 6):
            mk = notas == s
            if mk.any():
                pool = A["pool_coment"][s]
                j = r.integers(0, len(pool), int(mk.sum()))
                titulos[mk] = pool.review_comment_title.values[j]; msgs[mk] = pool.review_comment_message.values[j]
        avals = pd.DataFrame({"review_id": hexids(r, nr), "order_id": order_id[idx_r], "review_score": notas,
                              "review_comment_title": titulos, "review_comment_message": msgs,
                              "review_creation_date": fmt(cria[idx_r]), "review_answer_timestamp": fmt(resp[idx_r])})

        # ---------- SUJEIRA controlada ----------
        s = args.sujeira
        def marca(tipo_s, tabela, ids):
            for i in ids:
                gab.append((tipo_s, tabela, i, str(per)))
        if s > 0:
            mk = (r.random(n) < .002 * s) & ~pd.isna(aprov_o)
            ped_df.loc[mk, "order_approved_at"] = fmt(compra[mk] - r.integers(3600, 172800, int(mk.sum())).astype("timedelta64[s]")).values
            marca("aprovacao_antes_da_compra", "orders", order_id[mk])
            mk = r.random(n) < .003 * s
            alt = clientes.loc[mk, "customer_state"].map(lambda x: x.lower() + " ")
            clientes.loc[mk, "customer_state"] = alt
            marca("estado_inconsistente", "customers", customer_id[mk])
            mk = r.random(m) < .001 * s
            itens.loc[mk, "price"] = np.nan
            marca("preco_nulo", "order_items", itens.loc[mk, "order_id"].values)
            mk = r.random(nr) < .0005 * s
            avals.loc[mk, "review_score"] = r.choice([0, 6], size=int(mk.sum()))
            marca("nota_invalida", "order_reviews", avals.loc[mk, "order_id"].values)

        # chegada tardia e duplicatas: vao para o arquivo do mes seguinte
        prox_ok = (s > 0) and (mi + 1 < len(meses)) and (meses[mi + 1].start_time < stream_ini)
        tardio = np.zeros(n, bool)
        if prox_ok:
            tardio = r.random(n) < .01 * s
            ids_t = set(order_id[tardio].tolist())
            marca("chegada_tardia", "orders", order_id[tardio])
            sel_p = ped_df[tardio]
            novo_carry = {"orders": sel_p, "customers": clientes[tardio],
                          "order_items": itens[itens.order_id.isin(ids_t)],
                          "order_payments": pagamentos[pagamentos.order_id.isin(ids_t)],
                          "order_reviews": avals[avals.order_id.isin(ids_t)]}
            dup = r.random(n) < .005 * s
            novo_carry["orders"] = pd.concat([novo_carry["orders"], ped_df[dup & ~tardio]])
            marca("pedido_duplicado", "orders", order_id[dup & ~tardio])
            ped_df, clientes = ped_df[~tardio], clientes[~tardio]
            itens, pagamentos, avals = (itens[~itens.order_id.isin(ids_t)], pagamentos[~pagamentos.order_id.isin(ids_t)],
                                        avals[~avals.order_id.isin(ids_t)])
        else:
            novo_carry = {}

        # ---------- escrita ----------
        tabs = {"orders": ped_df, "customers": clientes, "order_items": itens,
                "order_payments": pagamentos, "order_reviews": avals}
        nomes = {"orders": "olist_orders_dataset", "customers": "olist_customers_dataset",
                 "order_items": "olist_order_items_dataset", "order_payments": "olist_order_payments_dataset",
                 "order_reviews": "olist_order_reviews_dataset"}
        chave_dia = lambda df, col: pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        data_pedido = pd.Series(pd.to_datetime(fmt(compra)).dt.normalize().values, index=order_id)
        for nome, df in tabs.items():
            if nome in carry and len(carry[nome]):
                df = pd.concat([carry[nome], df], ignore_index=True)
            if nome == "customers":
                d = data_pedido.reindex(df.customer_id.map(dict(zip(customer_id, order_id)))).values
            elif nome == "orders":
                d = pd.to_datetime(df.order_purchase_timestamp).dt.normalize().values
            else:
                d = data_pedido.reindex(df.order_id).values
            d = pd.Series(pd.to_datetime(d), index=df.index)
            em_stream = (d >= stream_ini) & d.notna() & (d == d)
            # historico: 1 arquivo por mes
            hist = df[~em_stream]
            if len(hist):
                pasta = os.path.join(saida, "fatos", nome); os.makedirs(pasta, exist_ok=True)
                hist.to_csv(os.path.join(pasta, f"{nomes[nome]}_{per}.csv"), index=False)
            if em_stream.any():
                pasta = os.path.join(saida, "stream", nome); os.makedirs(pasta, exist_ok=True)
                for dia, g in df[em_stream].groupby(d[em_stream].dt.strftime("%Y-%m-%d")):
                    g.to_csv(os.path.join(pasta, f"{nomes[nome]}_{dia}.csv"), index=False)
        carry = novo_carry

        resumo_meses.append({"mes": str(per), "pedidos": n, "itens": m,
                             "receita": float(np.nansum(preco) + np.nansum(frete)),
                             "pct_atraso": float(atraso[entregue].mean()) if entregue.any() else None,
                             "pct_pix": float((tipo == "pix").mean())})
        if mi % 12 == 0:
            print(f"      {per}  pedidos={n:,}  ({time.time() - t0:.0f}s)")

    print("[4/4] Gabarito e resumo ...")
    pd.DataFrame({"seller_id": cat["ven"].seller_id.values, "vendedor_ruim": cat["ruim"]}).to_csv(
        os.path.join(saida, "_gabarito", "vendedores_ruins.csv"), index=False)
    g = pd.DataFrame(gab, columns=["tipo_sujeira", "tabela", "chave", "particao"])
    g.to_csv(os.path.join(saida, "_gabarito", "sujeira_injetada.csv"), index=False)
    resumo = {"semente": args.semente, "pedidos": args.pedidos, "inicio": args.inicio, "fim": args.fim,
              "stream_dias": args.stream_dias, "sujeira": args.sujeira,
              "sujeira_por_tipo": g.tipo_sujeira.value_counts().to_dict(),
              "premissas": {"crescimento_anual": CRESCIMENTO_ANUAL, "ipca_anual": IPCA_ANUAL},
              "meses": resumo_meses, "tempo_segundos": round(time.time() - t0, 1)}
    with open(os.path.join(saida, "_gabarito", "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1, default=float)
    print(f"Pronto em {time.time() - t0:.0f}s. Saida: {saida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pedidos", type=int, default=2_000_000, help="total de pedidos sinteticos")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--inicio", default="2018-09-01")
    ap.add_argument("--fim", default=date.today().isoformat())
    ap.add_argument("--olist", default="dados/olist", help="pasta com os CSVs reais do Olist")
    ap.add_argument("--saida", default="saida")
    ap.add_argument("--stream-dias", type=int, default=30, help="ultimos N dias saem em arquivos diarios (pasta stream/)")
    ap.add_argument("--sujeira", type=float, default=1.0, help="multiplicador da sujeira injetada (0 desliga)")
    gerar(ap.parse_args())