#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score

from lightgbm import LGBMRanker
from catboost import CatBoostRanker, Pool

from rag.config import load_config
from rag.ranker_features import build_features_for_query
from sentence_transformers import CrossEncoder


def load_ranker_dataset(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(item)
    return data


def build_dataset(cfg, records):
    """
    records: list of {"query", "chunk", "label"}
    -> X, y, groups
    """
    # группируем по query
    from collections import defaultdict

    by_query = defaultdict(list)
    for r in records:
        by_query[r["query"]].append(r)

    rcfg = cfg.ranker
    ce_model = None
    if rcfg.use_cross_encoder_feature:
        ce_model = CrossEncoder(cfg.reranker.model_name)

    X_all = []
    y_all = []
    groups = []

    for qi, (q, items) in enumerate(by_query.items()):
        # candidates в формате pipeline'а
        candidates = []
        for pos, r in enumerate(items):
            candidates.append(
                {
                    "text": r["chunk"],
                    "retrieval_rank": pos,
                }
            )
        X_q, feature_names, _ = build_features_for_query(
            cfg, q, candidates, ce_model=ce_model
        )
        y_q = np.array([r["label"] for r in items], dtype="float32")

        X_all.append(X_q)
        y_all.append(y_q)
        groups.append(len(items))

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups_arr = np.array(groups, dtype=int)

    return X, y, groups_arr, feature_names


def train_ranker():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train supervised ranker (LightGBMRanker / CatBoostRanker)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к config.yaml",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ranker/train_ranker.jsonl",
        help="JSONL с (query, chunk, label)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["lgbm", "catboost"],
        help="Тип модели; если None — берём из config.ranker.model_type",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    rcfg = cfg.ranker
    model_type = args.model_type or rcfg.model_type

    records = load_ranker_dataset(args.data)
    X, y, groups, feature_names = build_dataset(cfg, records)

    print(f"📦 samples: {len(y)}, features: {X.shape[1]}, groups: {len(groups)}")

    # CV по группам (query)
    gkf = GroupKFold(n_splits=3)
    group_ids = []
    # разворачиваем groups → group_id на каждую строку
    gi = 0
    for n in groups:
        group_ids.extend([gi] * n)
        gi += 1
    group_ids = np.array(group_ids, dtype=int)

    ndcgs = []

    if model_type == "lgbm":
        print("🧠 Training LightGBMRanker...")
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=group_ids), 1):
            model = LGBMRanker(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="lambdarank",
                random_state=42 + fold,
            )
            # группировки для LightGBM — длины групп
            # получаем groups_tr/val как суммы по group_ids
            def make_lgbm_groups(idx):
                # idx — индексы строк, нужно получить длины групп в порядке их появления
                from collections import OrderedDict

                d = OrderedDict()
                for i in idx:
                    g = int(group_ids[i])
                    d.setdefault(g, 0)
                    d[g] += 1
                return list(d.values())

            grp_tr = make_lgbm_groups(tr_idx)
            grp_val = make_lgbm_groups(val_idx)

            model.fit(
                X[tr_idx],
                y[tr_idx],
                group=grp_tr,
                eval_set=[(X[val_idx], y[val_idx])],
                eval_group=[grp_val],
                eval_at=[cfg.index.top_k],
                verbose=False,
            )

            y_pred = model.predict(X[val_idx])
            # считаем NDCG@k на уровне fold
            # Для NDCG нужен разбор по группам
            ndcgs_fold = []
            start = 0
            for g_size in grp_val:
                end = start + g_size
                nd = ndcg_score(
                    [y[val_idx][start:end]],
                    [y_pred[start:end]],
                    k=cfg.index.top_k,
                )
                ndcgs_fold.append(nd)
                start = end
            ndcgs.append(float(np.mean(ndcgs_fold)))
            print(f"Fold {fold}: NDCG@{cfg.index.top_k} = {ndcgs[-1]:.4f}")

        print(f"✅ CV mean NDCG@{cfg.index.top_k}: {np.mean(ndcgs):.4f}")
        # Обучаем финальную модель на всём датасете
        final_model = LGBMRanker(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="lambdarank",
            random_state=42,
        )
        final_groups = groups.tolist()
        final_model.fit(X, y, group=final_groups)
        Path(rcfg.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, rcfg.model_path)
        print(f"💾 LightGBMRanker сохранён в {rcfg.model_path}")

    else:
        print("🧠 Training CatBoostRanker...")
        # для CatBoost группировка через group_id прямо
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=group_ids), 1):
            train_pool = Pool(
                X[tr_idx],
                y[tr_idx],
                group_id=group_ids[tr_idx],
                feature_names=feature_names,
            )
            val_pool = Pool(
                X[val_idx],
                y[val_idx],
                group_id=group_ids[val_idx],
                feature_names=feature_names,
            )
            model = CatBoostRanker(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="YetiRank",
                random_seed=42 + fold,
                verbose=False,
            )
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            y_pred = model.predict(val_pool)

            # NDCG по группам
            ndcgs_fold = []
            # получаем размеры групп для этого фолда
            from collections import OrderedDict
            d = OrderedDict()
            for g in group_ids[val_idx]:
                d.setdefault(int(g), 0)
                d[int(g)] += 1
            grp_val = list(d.values())
            start = 0
            for g_size in grp_val:
                end = start + g_size
                nd = ndcg_score(
                    [y[val_idx][start:end]],
                    [y_pred[start:end]],
                    k=cfg.index.top_k,
                )
                ndcgs_fold.append(nd)
                start = end
            ndcgs.append(float(np.mean(ndcgs_fold)))
            print(f"Fold {fold}: NDCG@{cfg.index.top_k} = {ndcgs[-1]:.4f}")

        print(f"✅ CV mean NDCG@{cfg.index.top_k}: {np.mean(ndcgs):.4f}")
        final_pool = Pool(
            X,
            y,
            group_id=group_ids,
            feature_names=feature_names,
        )
        final_model = CatBoostRanker(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="YetiRank",
            random_seed=42,
            verbose=False,
        )
        final_model.fit(final_pool, verbose=False)
        Path(rcfg.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, rcfg.model_path)
        print(f"💾 CatBoostRanker сохранён в {rcfg.model_path}")


if __name__ == "__main__":
    train_ranker()