"""Generate the fully synthetic demo dataset in data/demo/.

Everything here is invented: no real judgments, no real people. The reference
labels are CONSTRUCTED from the template sentences that were inserted; they are
not GPT-4o labels and carry no information about the paper's results. A fact section has a
variable number of differently typed parts in random order (narrative parts that carry the
factor information, and boilerplate such as procedure, evidence, case origin or other notes),
so chunking separates the types and the CUA has something to filter. The 40
CUA passages are synthetic as well (20 per class).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CASE_TYPE = "demo_theft"

FACTORS = [
    "攜帶兇器行竊",
    "夜間侵入住宅竊取財物",
    "被告於案發後自首",
    "被告與被害人達成和解並賠償",
    "被告為累犯，前案執行完畢後再犯",
]

FACTOR_SENTENCES = [
    ["被告隨身攜帶水果刀一把，持刀進入商店行竊。",
     "被告攜帶扳手作為犯案工具，該工具客觀上足以對人之生命身體構成威脅。"],
    ["被告於凌晨二時許，趁屋主熟睡之際侵入住宅，竊取現金。",
     "被告於夜間翻越圍牆進入他人住宅內，竊取電腦一台。"],
    ["案發後被告主動前往警局向員警坦承犯行並自首。",
     "被告於警方尚未查獲前，即主動投案自首並接受裁判。"],
    ["被告於偵查中與被害人達成和解，並已賠償全部損失。",
     "被告事後以新臺幣三萬元與被害人和解，被害人表示願意原諒被告。"],
    ["被告前因竊盜案件經法院判處有期徒刑六月確定，於執行完畢後二年內再犯本案。",
     "被告五年內曾因竊盜罪受有期徒刑執行完畢，仍再次竊取他人財物。"],
]
FACTOR_PROB = [0.30, 0.30, 0.30, 0.30, 0.25]

NARRATIVE = [
    "甲於某日下午行經某處停車場。",
    "甲見一輛未上鎖之腳踏車停放路旁，一時貪念將之竊取。",
    "甲將竊得之物品變賣，所得款項花用殆盡。",
    "被害人發現財物遭竊後報警處理。",
    "警方調閱監視器畫面後鎖定甲。",
    "甲在店內徘徊，趁店員不注意拿取商品放入背包。",
    "甲空手進入店內，趁無人注意拿取貨架上之商品。",
    "甲離開現場後，將所竊商品交予不知情之友人。",
    "被害人事後清點，發現短少財物價值約數千元。",
    "甲行竊時身穿深色外套並戴口罩。",
]

PROCEDURE = [
    "本案經檢察官提起公訴，並由本院依通常程序審理。",
    "被告於審理中對起訴書所載犯罪事實均坦承不諱。",
    "公訴人及被告均到庭陳述意見。",
    "本案經檢察官到庭執行職務。",
    "本院審理時，被告及辯護人均未聲請調查其他證據。",
]
EVIDENCE = [
    "以上事實，有下列證據可資佐證：被害人之警詢筆錄、現場監視器錄影畫面截圖及扣案物品照片。",
    "上開證據均經本院於審理時依法調查。",
    "此外並有贓物認領保管單及現場照片附卷可稽。",
    "上開事實，亦有證物採證報告在卷可佐。",
    "上開各項證據，均與被害人所述情節相符。",
]


ORIGIN = [
    "案經某市警察局報請某地方檢察署檢察官偵查起訴。",
    "案經被害人向某分局報案，由某地方檢察署檢察官偵查起訴。",
    "案經某縣警察局移送某地方檢察署檢察官偵查起訴。",
]
OTHER = [
    "被告另涉其他案件，業經另案審理，不在本案審理範圍。",
    "被害人未於本案提起附帶民事訴訟。",
    "本案起訴書所載之部分事項，業經檢察官於審理時更正。",
    "本案原定之審理期日，經合法通知後延期。",
]
BOILERPLATE_POOLS = [PROCEDURE, EVIDENCE, ORIGIN, OTHER]


def narrative_sections(rng, flags):
    """1-3 narrative sections; every present factor sentence goes into one of them."""
    n = rng.randint(1, 3)
    secs = [rng.sample(NARRATIVE, rng.randint(2, 4)) for _ in range(n)]
    for j, on in enumerate(flags):
        if on:
            secs[rng.randrange(n)].append(rng.choice(FACTOR_SENTENCES[j]))
    for sec in secs:
        rng.shuffle(sec)
    return ["".join(sec) for sec in secs]


def fact_sections(rng, flags):
    """A fact section made of a variable number of differently typed parts in random order:
    narrative (carries the factor information) and boilerplate of several kinds
    (procedure, evidence, case origin, other). Not every kind occurs in every judgment."""
    parts = narrative_sections(rng, flags)
    kinds = rng.sample(range(len(BOILERPLATE_POOLS)), rng.randint(1, len(BOILERPLATE_POOLS)))
    for k in kinds:
        parts.append(boilerplate(rng, BOILERPLATE_POOLS[k], 1, 2))
    rng.shuffle(parts)
    return "".join(f"{HEADINGS[i]}、{p}" for i, p in enumerate(parts))


HEADINGS = "一二三四五六七八九十"


def boilerplate(rng, pool, lo=2, hi=3):
    return "".join(rng.sample(pool, rng.randint(lo, hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--n-sft", type=int, default=640, help="judgments reserved for building the fine-tuning data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    judgments, labels = [], []
    for i in range(args.n):
        flags = [rng.random() < p for p in FACTOR_PROB]
        jid = f"demo-{i + 1:03d}"
        judgments.append({"id": jid, "case_type": CASE_TYPE, "fact": fact_sections(rng, flags)})
        labels.append({"id": jid, "labels": {f: int(on) for f, on in zip(FACTORS, flags)}})

    passages = []
    for _ in range(20):
        passages.append({"text": rng.choice(NARRATIVE) + "".join(rng.sample(NARRATIVE, 2))
                         + rng.choice(rng.choice(FACTOR_SENTENCES)), "label": "important"})
    for _ in range(20):
        pool = rng.choice(BOILERPLATE_POOLS)
        passages.append({"text": boilerplate(rng, pool, 1, 3), "label": "unimportant"})
    rng.shuffle(passages)

    def dump(name, rows):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("judgments.jsonl", judgments)
    dump("reference_labels.jsonl", labels)
    dump("cua_passages.jsonl", passages)
    (out / "factors.json").write_text(json.dumps({CASE_TYPE: FACTORS}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Disjoint judgment-level split: "train" is the only pool fine-tuning data may come from,
    # "test" is the pool all methods are evaluated on.
    ids = [j["id"] for j in judgments]
    (out / "splits").mkdir(exist_ok=True)
    (out / "splits" / "sft.json").write_text(
        json.dumps({"train": ids[: args.n_sft], "test": ids[args.n_sft:]}, indent=2), encoding="utf-8")
    print(f"wrote {len(judgments)} judgments, {len(passages)} CUA passages to {out}")


if __name__ == "__main__":
    main()
