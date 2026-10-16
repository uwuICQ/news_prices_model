"""Дашборд: лента новостей, график, сезонность и прогноз."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.db import DB_PATH, connect
from src.features import adjust_price_scale, next_trading_day, seasonal_profile
from src.universe import Universe, load_universe

MONTHS = ["", "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
FEATURE_LABELS = {
    "sent_mean_alias": "Средняя тональность по компании",
    "sent_sum_alias": "Сумма тональности по компании",
    "news_count_alias": "Число новостей по компании",
    "sent_mean_sector": "Средняя тональность сегмента",
    "sent_sum_sector": "Сумма тональности сегмента",
    "news_count_sector": "Число новостей сегмента",
    "sent_mean_macro": "Средняя макро-тональность",
    "news_count_macro": "Число макроновостей",
    "ret_1": "Доходность 1 день",
    "ret_5": "Доходность 5 дней",
    "ret_20": "Доходность 20 дней",
    "volume_z": "Объём к норме 20 дней",
    "month_sin": "Месяц (sin)",
    "month_cos": "Месяц (cos)",
    "dow_sin": "День недели (sin)",
    "dow_cos": "День недели (cos)",
    "ticker": "Бумага",
}


def _query(sql: str) -> pd.DataFrame:
    conn = connect()
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    return adjust_price_scale(_query("SELECT * FROM prices"))


@st.cache_data(show_spinner=False)
def load_news() -> pd.DataFrame:
    return _query("SELECT channel, msg_id, published_at, text, sentiment, relevant, is_macro FROM news")


@st.cache_data(show_spinner=False)
def load_links() -> pd.DataFrame:
    return _query("SELECT * FROM news_links")


@st.cache_data(show_spinner=False)
def load_status() -> pd.DataFrame:
    return _query("SELECT * FROM ticker_status")


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return _query("SELECT name, value, detail FROM model_metrics")


@st.cache_data(show_spinner=False)
def load_importance() -> pd.DataFrame:
    return _query("SELECT feature, importance FROM feature_importance")


def _tickers_for_sector(universe: Universe, sector_id: str) -> list:
    return [item for item in universe.tickers if item.sector == sector_id]


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:+.2f}%"


def _forecast_table(forecasts: list[dict]) -> None:
    if not forecasts:
        return
    frame = pd.DataFrame(forecasts)
    view = pd.DataFrame(
        {
            "Бумага": frame["ticker"] + " · " + frame["name"],
            "Сегмент": frame["sector"],
            "Направление": frame["direction"],
            "Ожидаемая доходность": frame["expected_return"].map(_fmt_pct),
            "Тональность": frame["sentiment"].map(lambda value: f"{value:+.2f}"),
            "Привязка": frame["reason"],
            "Комментарий": frame["note"],
        }
    )
    st.dataframe(view, hide_index=True, width="stretch")


def _render_status(status: pd.DataFrame, universe: Universe) -> None:
    if status.empty:
        return
    failed = status.loc[status["ok"] != 1]
    if failed.empty:
        return
    names = []
    for row in failed.itertuples(index=False):
        try:
            title = universe.get(row.ticker).name
        except KeyError:
            title = row.ticker
        names.append(f"{row.ticker} ({title}): {row.note or 'нет данных'}")
    st.warning("Биржа не отдала полный ряд: " + "; ".join(names))


def tab_feed(universe: Universe, prices: pd.DataFrame, news: pd.DataFrame, links: pd.DataFrame) -> None:
    sectors = list(universe.sectors.items())
    sector_id = st.selectbox("Сегмент", options=[key for key, _ in sectors], format_func=lambda key: universe.sector_name(key))
    choices = _tickers_for_sector(universe, sector_id)
    ticker = st.selectbox("Бумага", options=[item.ticker for item in choices], format_func=lambda code: f"{code} · {universe.get(code).name}")
    info = universe.get(ticker)
    if info.history_note:
        st.caption(info.history_note)
    channels = ["оба"] + list(universe.channels)
    channel = st.selectbox("Канал", options=channels, format_func=lambda value: {"rian_ru": "РИА Новости", "mash": "Mash"}.get(value, "РИА и Mash"))
    series = prices.loc[prices["ticker"] == ticker].sort_values("trade_date")
    if series.empty:
        st.info("Для этой бумаги в базе нет цен. Запустите python -m src.collect_prices.")
        return
    last_day = date.fromisoformat(str(series["trade_date"].iloc[-1]))
    first_day = date.fromisoformat(str(series["trade_date"].iloc[0]))
    default_start = max(first_day, last_day - timedelta(days=180))
    picked = st.date_input("Период", value=(default_start, last_day), min_value=first_day, max_value=last_day)
    if not isinstance(picked, tuple) or len(picked) != 2:
        st.info("Выберите начало и конец периода.")
        return
    start, end = picked
    window = series.loc[(series["trade_date"] >= start.isoformat()) & (series["trade_date"] <= end.isoformat())]
    calendar = series["trade_date"].astype(str).tolist()
    linked = pd.DataFrame()
    if not news.empty and not links.empty:
        ticker_links = links.loc[links["ticker"] == ticker]
        linked = ticker_links.merge(news, on=["channel", "msg_id"], how="left")
        if channel != "оба":
            linked = linked.loc[linked["channel"] == channel]
        linked = linked.copy()
        linked["day"] = linked["published_at"].astype(str).str.slice(0, 10)
        linked = linked.loc[(linked["day"] >= start.isoformat()) & (linked["day"] <= end.isoformat())]
        linked["trade_date"] = [next_trading_day(calendar, day) for day in linked["day"]]
        linked = linked.dropna(subset=["trade_date"])
        returns = series[["trade_date", "close"]].copy()
        returns["next_ret"] = returns["close"].shift(-1) / returns["close"] - 1
        linked = linked.merge(returns[["trade_date", "next_ret", "close"]], on="trade_date", how="left")
    elif news.empty:
        st.caption("Новостей в базе нет. Соберите их: python -m src.collect_news, затем python -m src.train.")
    else:
        st.caption("Новости ещё не привязаны к бумагам. Выполните python -m src.train.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["trade_date"], y=window["close"], mode="lines", name="Закрытие", line={"color": "#1f4b99"}))
    if not linked.empty:
        markers = linked.groupby("trade_date", as_index=False).agg(close=("close", "first"), sentiment=("sentiment", "mean"), n=("msg_id", "count"))
        markers["sentiment"] = markers["sentiment"].fillna(0.0)
        fig.add_trace(
            go.Scatter(
                x=markers["trade_date"],
                y=markers["close"],
                mode="markers",
                name="Новости",
                marker={"size": 9, "color": markers["sentiment"], "colorscale": "RdYlGn", "cmin": -1, "cmax": 1, "showscale": True, "colorbar": {"title": "тон"}},
                text=[f"{int(n)} новости, тон {sent:+.2f}" for n, sent in zip(markers["n"], markers["sentiment"])],
                hovertemplate="%{x}<br>%{text}<extra></extra>",
            )
        )
    fig.update_layout(margin={"l": 8, "r": 8, "t": 8, "b": 8}, height=420, legend={"orientation": "h"})
    st.plotly_chart(fig, width="stretch")

    if linked.empty:
        return
    st.subheader("Лента")
    show_n = st.slider("Сколько постов показать", min_value=1, max_value=min(200, len(linked)), value=min(30, len(linked)))
    view = linked.sort_values("published_at", ascending=False).head(int(show_n))
    for row in view.itertuples(index=False):
        tone = "—" if pd.isna(row.sentiment) else f"{row.sentiment:+.2f}"
        kind = "алиас" if row.match_type == "alias" else "сегмент"
        source = "РИА Новости" if row.channel == "rian_ru" else "Mash"
        st.markdown(f"**{str(row.published_at)[:16]}** · {source} · {kind}: {row.match_term} · тон {tone} · след. день {_fmt_pct(row.next_ret)}")
        st.text(str(row.text)[:700])


def tab_season(universe: Universe, prices: pd.DataFrame) -> None:
    sector_id = st.selectbox(
        "Сегмент",
        options=list(universe.sectors),
        format_func=lambda key: universe.sector_name(key),
        key="season_sector",
    )
    if prices.empty:
        st.info("Нет цен.")
        return
    first = prices.groupby("ticker")["trade_date"].min()
    limit = (date.fromisoformat(universe.start) + timedelta(days=31)).isoformat()
    chosen = []
    skipped = []
    for item in _tickers_for_sector(universe, sector_id):
        if not item.seasonal:
            skipped.append(f"{item.ticker}: {item.history_note or 'короткий ряд'}")
            continue
        start = first.get(item.ticker)
        if start is None or str(start) > limit:
            skipped.append(f"{item.ticker}: ряд короче трёх лет")
            continue
        chosen.append(item.ticker)
    if skipped:
        st.caption("Вне сезонного профиля: " + "; ".join(skipped))
    if not chosen:
        st.info("В этом сегменте нет бумаг с рядом на всё трёхлетнее окно.")
        return
    profile = seasonal_profile(prices, chosen)
    profile["month_name"] = profile["month"].map(lambda month: MONTHS[int(month)])
    profile["paper"] = profile["ticker"].map(lambda code: f"{code} · {universe.get(code).name}")
    profile["mean_pct"] = profile["mean_return"] * 100
    fig = px.bar(profile, x="month_name", y="mean_pct", color="paper", barmode="group", labels={"month_name": "Месяц", "mean_pct": "Средняя дневная доходность, %", "paper": "Бумага"})
    fig.update_layout(margin={"l": 8, "r": 8, "t": 8, "b": 8}, height=460, xaxis={"categoryorder": "array", "categoryarray": MONTHS[1:]})
    st.plotly_chart(fig, width="stretch")
    st.caption("Средняя доходность закрытие-к-закрытию по календарным месяцам за доступные три года. Это описание сезонности, не прогноз.")


def tab_forecast(universe: Universe) -> None:
    st.write("Прогноз строится только по бумагам, которые нашлись в тексте. Если совпадений нет, оценка «на весь рынок» не делается.")
    text = st.text_area("Текст новости", height=160, placeholder="Вставьте новость. Например: Сбербанк повысил прогноз по прибыли, а цены на нефть снизились.")
    channel = st.selectbox("Канал для последних постов", options=list(universe.channels), format_func=lambda value: "РИА Новости" if value == "rian_ru" else "Mash", key="recent_channel")
    col_paste, col_recent = st.columns(2)
    run_paste = col_paste.button("Прогноз по этому тексту", type="primary")
    run_recent = col_recent.button("Последние посты из канала")
    if not run_paste and not run_recent:
        return
    try:
        from src.predict import forecast_texts, recent_news
    except Exception as exc:
        st.error(f"Не удалось загрузить модель прогноза: {exc}")
        return
    try:
        if run_paste:
            if not text.strip():
                st.warning("Вставьте текст новости.")
                return
            with st.spinner("Считаю тональность и прогноз"):
                results = forecast_texts([{"text": text.strip()}])
        else:
            items = recent_news(channel, limit=10)
            if not items:
                st.info("В базе нет постов этого канала. Сначала: python -m src.collect_news")
                return
            with st.spinner("Считаю тональность и прогноз"):
                results = forecast_texts(items)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Прогноз не посчитался: {exc}")
        return
    for result in results:
        if result.get("published_at"):
            source = "РИА Новости" if result.get("channel") == "rian_ru" else "Mash"
            st.markdown(f"**{str(result['published_at'])[:16]}** · {source}")
            st.text(result["text"][:500])
        if result["unlinked"]:
            st.info("Новость не привязана к бумагам из вселенной. Прогноз по всем бумагам сразу не строится.")
            continue
        _forecast_table(result["forecasts"])


def tab_metrics() -> None:
    metrics = load_metrics()
    if metrics.empty:
        st.info("Модель ещё не обучена. После сбора новостей выполните python -m src.train.")
        return
    values = {row.name: row.value for row in metrics.itertuples(index=False)}
    summary = metrics.loc[metrics["name"] == "summary", "detail"]
    if not summary.empty and summary.iloc[0]:
        st.write(summary.iloc[0])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE модели, п.п.", f"{values.get('mae', float('nan')) * 100:.3f}")
    c2.metric("MAE прогноза «0», п.п.", f"{values.get('mae_zero', float('nan')) * 100:.3f}")
    c3.metric("Точность знака", f"{values.get('dir_acc', float('nan')) * 100:.1f}%")
    c4.metric("Знак как вчера", f"{values.get('dir_yesterday', float('nan')) * 100:.1f}%")
    st.caption(f"Строк в обучении: {int(values.get('n_train', 0))}. Строк в проверке (последний год): {int(values.get('n_test', 0))}.")
    importance = load_importance()
    if importance.empty:
        return
    importance = importance.sort_values("importance", ascending=False)
    importance["label"] = importance["feature"].map(lambda name: FEATURE_LABELS.get(name, name))
    fig = px.bar(importance, x="importance", y="label", orientation="h", labels={"importance": "Важность (gain)", "label": ""})
    fig.update_layout(margin={"l": 8, "r": 8, "t": 8, "b": 8}, height=480, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Новости и акции Мосбиржи", layout="wide")
    st.title("Прогноз акций Мосбиржи по новостям")
    st.caption("Учебная модель: ожидаемая доходность следующего торгового дня по новостям РИА и Mash. Не является инвестиционной рекомендацией.")
    if st.button("Обновить данные из базы"):
        st.cache_data.clear()
    if not DB_PATH.exists():
        st.info("Базы ещё нет. Сначала соберите цены: python -m src.collect_prices")
        return
    universe = load_universe()
    prices = load_prices()
    news = load_news()
    links = load_links()
    _render_status(load_status(), universe)
    feed, season, forecast, metrics = st.tabs(["Лента", "Сезонность", "Новая новость", "Проверка"])
    with feed:
        tab_feed(universe, prices, news, links)
    with season:
        tab_season(universe, prices)
    with forecast:
        tab_forecast(universe)
    with metrics:
        tab_metrics()


main()
