import streamlit as st
import pandas as pd
from kafka import KafkaProducer
import json
import time
import os
import uuid

import altair as alt
import psycopg2
import requests

# Конфигурация Kafka
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions")
}

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://app:app@postgres:5432/fraud")
FRAUD_API_URL = os.getenv("FRAUD_API_URL", "http://fraud_detector:8000")

LAST_N = 100

def load_file(uploaded_file):
    """Загрузка CSV файла в DataFrame"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка данных в Kafka с уникальным ID транзакции"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            security_protocol="PLAINTEXT"
        )
        
        # Генерация уникальных ID для всех транзакций
        df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            # Отправляем данные вместе с ID
            producer.send(
                topic, 
                value={
                    "transaction_id": row['transaction_id'],
                    "data": row.drop('transaction_id').to_dict()
                }
            )
            progress_bar.progress((idx + 1) / total_rows)
            time.sleep(0.01)
            
        producer.flush()

        # Запоминаем ID последней отправленной пачки для раздела с результатами
        st.session_state.last_batch_ids = df['transaction_id'].tolist()

        return True
    except Exception as e:
        st.error(f"Ошибка отправки данных: {str(e)}")
        return False

def query_db(sql, params=None):
    """Выполнить запрос к Postgres и вернуть DataFrame"""
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [c[0] for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)
    finally:
        conn.close()


def show_histogram(scores, title):
    st.markdown(title)
    chart = alt.Chart(scores).mark_bar().encode(
        alt.X("score:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="score"),
        alt.Y("count()", title="Количество транзакций"),
    )
    st.altair_chart(chart, use_container_width=True)


def current_threshold():
    """Порог активной модели из API (None, если сервис недоступен)"""
    try:
        state = requests.get(FRAUD_API_URL + "/models", timeout=10).json()
        return next(m["threshold"] for m in state["models"] if m["id"] == state["active_id"])
    except Exception:
        return None


def show_results():
    """Раздел «Посмотреть результаты»: данные из Postgres"""
    # 1. Последние 10 транзакций, у которых score больше текущего порога.
    # Фрод определяется по текущему порогу, а не по флагу, сохранённому при скоринге,
    # поэтому изменение порога сразу меняет список (в Kafka и в БД лежит исходный fraud_flag).
    threshold = current_threshold()
    if threshold is None:
        st.warning("Не удалось получить текущий порог, использую сохранённый fraud_flag")
        title, where, params = "**10 последних транзакций с fraud_flag = 1**", "fraud_flag = 1", None
    else:
        title = f"**10 последних транзакций со score > {threshold:.2f}** (текущий порог)"
        where, params = "score > %s", (threshold,)
    st.markdown(title)
    fraud = query_db(
        f"SELECT transaction_id, score, fraud_flag, created_at FROM scores "
        f"WHERE {where} ORDER BY created_at DESC LIMIT 10", params
    )
    if fraud.empty:
        st.info("Транзакций выше порога пока нет")
    else:
        st.dataframe(fraud, use_container_width=True)

    # 2. Гистограмма скоров (выбор режима стоит после таблицы)
    mode = st.radio("Гистограмма скоров по:", [MODE_LAST_N, MODE_LAST_BATCH])
    if mode == MODE_LAST_N:
        scores = query_db(f"SELECT score FROM scores ORDER BY created_at DESC LIMIT {LAST_N}")
        if scores.empty:
            st.info("В базе пока нет транзакций")
        else:
            show_histogram(scores, f"**Распределение скоров последних {len(scores)} транзакций**")
    else:
        ids = st.session_state.get("last_batch_ids")
        if not ids:
            st.info("В этой сессии ещё ничего не отправляли. Отправьте файл или выберите другой режим.")
        else:
            scores = query_db("SELECT score FROM scores WHERE transaction_id = ANY(%s)", (ids,))
            if scores.empty:
                st.info("Скоры отправленной пачки ещё не появились в базе, попробуйте через несколько секунд")
            else:
                show_histogram(
                    scores,
                    f"**Распределение скоров последней отправленной пачки** "
                    f"(в базе {len(scores)} из {len(ids)} транзакций)"
                )


# Инициализация состояния
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Интерфейс
st.title("📤 Отправка данных в Kafka")

# Блок загрузки файлов
uploaded_file = st.file_uploader(
    "Загрузите CSV файл с транзакциями",
    type=["csv"]
)

if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    # Добавляем файл в состояние
    st.session_state.uploaded_files[uploaded_file.name] = {
        "status": "Загружен",
        "df": load_file(uploaded_file)
    }
    st.success(f"Файл {uploaded_file.name} успешно загружен!")

# Список загруженных файлов
if st.session_state.uploaded_files:
    st.subheader("🗂 Список загруженных файлов")
    
    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])
        
        with cols[0]:
            st.markdown(f"**Файл:** `{file_name}`")
            st.markdown(f"**Статус:** `{file_data['status']}`")
        
        with cols[2]:
            if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                if file_data["df"] is not None:
                    with st.spinner("Отправка..."):
                        success = send_to_kafka(
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"]
                        )
                        if success:
                            st.session_state.uploaded_files[file_name]["status"] = "Отправлен"
                            st.rerun()
                else:
                    st.error("Файл не содержит данных")

# Раздел с результатами скоринга
st.divider()
st.header("📊 Результаты скоринга")

MODE_LAST_N = f"Последние {LAST_N} транзакций"
MODE_LAST_BATCH = "Вся последняя отправленная пачка"

if st.button("Посмотреть результаты"):
    st.session_state.show_results = True

if st.session_state.get("show_results"):
    try:
        show_results()
    except Exception as e:
        st.error(f"Не удалось получить данные из Postgres: {str(e)}")


# ---------------------------------------------------------------------------
# Модель: переобучение, порог, история (запросы идут в API сервиса fraud_detector)
# ---------------------------------------------------------------------------
SAMPLE_SIZES = ["100", "1 000", "10 000", "50 000", "Весь train.csv"]
METRIC_COLS = ["roc_auc", "pr_auc", "best_f1", "best_f1_threshold"]


def api(method, path, **kwargs):
    """Вызов API; при ошибке показывает сообщение и возвращает None"""
    try:
        resp = requests.request(method, FRAUD_API_URL + path, timeout=kwargs.pop("timeout", 30), **kwargs)
    except requests.RequestException as e:
        st.error(f"Сервис скоринга недоступен: {e}")
        return None
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"Ошибка {resp.status_code}: {detail}")
        return None
    return resp.json()


def show_metrics(m):
    cols = st.columns(4)
    cols[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    cols[1].metric("PR-AUC", f"{m['pr_auc']:.3f}")
    cols[2].metric("Лучший F1", f"{m['best_f1']:.3f}")
    cols[3].metric("Порог лучшего F1", f"{m['best_f1_threshold']:.3f}")


def history_table(models, active_id):
    rows = []
    for m in models:
        rows.append({
            "id": m["id"],
            "активна": "✅" if m["id"] == active_id else "",
            "тип": m["name"] + (" (дефолт)" if m["is_default"] else ""),
            "создана": m["created_at"][:19].replace("T", " "),
            "C": m["params"]["C"],
            "class_weight": m["params"]["class_weight"],
            "max_iter": m["params"]["max_iter"],
            "порог": m["threshold"],
            **{k: round(m["metrics"][k], 4) for k in METRIC_COLS},
        })
    return pd.DataFrame(rows)


st.divider()
st.header("🧠 Модель")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

state = api("GET", "/models")
if state:
    models, active_id = state["models"], state["active_id"]
    active = next(m for m in models if m["id"] == active_id)
    default = next(m for m in models if m["is_default"])

    st.markdown(
        f"**Активная модель:** #{active['id']} ({active['name']}), логистическая регрессия, "
        f"C={active['params']['C']}, class_weight={active['params']['class_weight']}, "
        f"max_iter={active['params']['max_iter']}"
    )
    show_metrics(active["metrics"])
    st.caption("Метрики посчитаны на отложенной выборке (20% train.csv). Доля фрода в train.csv: "
               f"{active['metrics']['fraud_rate']:.2%}.")

    # Порог
    st.subheader("Порог фрода")
    threshold = st.slider("fraud_flag = 1, если score больше порога", 0.0, 1.0,
                          float(active["threshold"]), 0.01, key=f"threshold_{active_id}_{active['threshold']}")
    if st.button("Применить порог"):
        if api("POST", "/threshold", json={"value": threshold}):
            st.success(f"Порог {threshold:.2f} применён к модели #{active_id}")
            st.rerun()

    # Переобучение
    st.subheader("Переобучение")
    with st.form("retrain_form"):
        c1, c2, c3 = st.columns(3)
        C = c1.number_input("C (обратная сила регуляризации)", min_value=0.0001, max_value=10000.0,
                            value=float(default["params"]["C"]), format="%g")
        class_weight = c2.selectbox("class_weight", ["balanced", "none"],
                                    index=0 if default["params"]["class_weight"] == "balanced" else 1)
        max_iter = c3.number_input("max_iter", min_value=10, max_value=5000,
                                   value=int(default["params"]["max_iter"]), step=100)
        # Заглушка: пока обучаем только на всём train.csv
        st.selectbox("Количество последних примеров для обучения", SAMPLE_SIZES,
                     index=len(SAMPLE_SIZES) - 1, disabled=True,
                     help="Пока недоступно: обучение идёт на всём train.csv")
        submitted = st.form_submit_button("Переобучить модель")
    if submitted:
        with st.spinner("Обучение..."):
            row = api("POST", "/train", timeout=600,
                      json={"C": C, "class_weight": class_weight, "max_iter": int(max_iter)})
        if row:
            st.session_state.flash = f"Модель #{row['id']} обучена за {row['metrics']['fit_seconds']} с и стала активной"
            st.rerun()

    # История
    st.subheader("История моделей")
    st.dataframe(history_table(models, active_id), use_container_width=True, hide_index=True)
    c1, c2 = st.columns([2, 2])
    with c1:
        pick = st.selectbox("Модель из истории", [m["id"] for m in models])
        if st.button("Сделать активной"):
            if api("POST", f"/models/{pick}/activate"):
                st.rerun()
    with c2:
        st.write("")
        st.write("")
        if st.button(f"Сбросить на дефолтную модель (#{default['id']})"):
            if api("POST", "/models/reset-default"):
                st.rerun()
