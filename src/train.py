from __future__ import annotations

import sys
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.db import MODEL_PATH, connect
from src.features import FEATURE_COLS, build_panel, load_frames, prepare_links, seasonal_profile
from src.sentiment import score_pending
from src.universe import load_universe


def _setup_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def _direction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.sign(predicted[mask]) == np.sign(actual[mask])))


def _fit(panel: pd.DataFrame, train_end: str) -> dict:
    train = panel.loc[panel["trade_date"] < train_end].copy()
    test = panel.loc[panel["trade_date"] >= train_end].copy()
    if train.empty or test.empty:
        raise SystemExit(
            "Не хватает строк для разбиения на два года обучения и год проверки. "
            "Проверьте, что в базе есть цены и привязанные новости за всё окно."
        )
    categories = sorted(panel["ticker"].astype(str).unique())
    columns = FEATURE_COLS + ["ticker"]
    x_train = train[columns].copy()
    x_test = test[columns].copy()
    x_train["ticker"] = pd.Categorical(x_train["ticker"], categories=categories)
    x_test["ticker"] = pd.Categorical(x_test["ticker"], categories=categories)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, train["target"], categorical_feature=["ticker"])
    predicted = model.predict(x_test)
    actual = test["target"].to_numpy()
    mae = float(mean_absolute_error(actual, predicted))
    mae_zero = float(mean_absolute_error(actual, np.zeros_like(actual)))
    dir_model = _direction_accuracy(actual, predicted)
    dir_yesterday = _direction_accuracy(actual, test["ret_1"].to_numpy())
    if mae < mae_zero:
        summary = "На последнем годе модель ближе к факту, чем прогноз «доходность 0»."
    else:
        summary = "На последнем годе модель не обыграла прогноз «доходность 0». Связь новостей с ценой на этом окне слабая."
    gains = model.booster_.feature_importance(importance_type="gain")
    names = model.booster_.feature_name()
    importance = {name: float(value) for name, value in zip(names, gains)}
    return {
        "model": model,
        "ticker_categories": categories,
        "metrics": {
            "mae": mae,
            "mae_zero": mae_zero,
            "dir_acc": dir_model,
            "dir_yesterday": dir_yesterday,
            "n_train": float(len(train)),
            "n_test": float(len(test)),
        },
        "summary": summary,
        "importance": importance,
    }


def _save_metrics(conn, result: dict) -> None:
    conn.execute("DELETE FROM model_metrics")
    conn.execute("DELETE FROM feature_importance")
    rows = [(name, value, None) for name, value in result["metrics"].items()]
    rows.append(("summary", None, result["summary"]))
    conn.executemany("INSERT INTO model_metrics (name, value, detail) VALUES (?, ?, ?)", rows)
    conn.executemany(
        "INSERT INTO feature_importance (feature, importance) VALUES (?, ?)",
        list(result["importance"].items()),
    )
    conn.commit()


def main() -> None:
    _setup_stdout()
    universe = load_universe()
    conn = connect()
    try:
        print("привязка новостей", flush=True)
        relevant, links = prepare_links(conn, universe)
        print(f"  релевантных постов: {relevant}, связей: {links}", flush=True)
        if relevant == 0:
            raise SystemExit("Ни один пост не совпал с бумагами, сегментами или макрословами.")
        print("тональность", flush=True)
        try:
            scored = score_pending(conn)
        except Exception as exc:
            raise SystemExit(
                "Не удалось посчитать тональность. Нужна модель "
                "seara/rubert-tiny2-russian-sentiment: она скачается при первом запуске. "
                f"Ошибка: {exc}"
            ) from exc
        print(f"  новых оценок: {scored}", flush=True)
        prices, news, link_rows = load_frames(conn)
        if prices.empty:
            raise SystemExit("Нет цен. Сначала выполните: python -m src.collect_prices")
        panel = build_panel(prices, news, link_rows)
        if panel.empty:
            raise SystemExit("Панель пустая: не хватает истории цен для признаков за 20 дней.")
        print(f"панель: {len(panel)} строк", flush=True)
        seasonal_names = [item.ticker for item in universe.tickers if item.seasonal]
        profile = seasonal_profile(prices, seasonal_names)
        if profile.empty:
            print("сезонный профиль пуст: нет рядов с полной историей", flush=True)
        else:
            print(f"сезонный профиль: {profile['ticker'].nunique()} бумаг", flush=True)
        result = _fit(panel, universe.train_end)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": result["model"],
                "ticker_categories": result["ticker_categories"],
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_end": universe.train_end,
            },
            MODEL_PATH,
        )
        _save_metrics(conn, result)
        metrics = result["metrics"]
        print(
            f"MAE модели {metrics['mae']:.5f} | MAE нуля {metrics['mae_zero']:.5f} | "
            f"знак {metrics['dir_acc']:.3f} | знак вчера {metrics['dir_yesterday']:.3f}",
            flush=True,
        )
        print(result["summary"], flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
