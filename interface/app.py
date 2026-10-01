import streamlit as st
import pandas as pd
from kafka import KafkaProducer
import json
import time
import os
import uuid

import altair as alt
import psycopg2

# Конфигурация Kafka
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions")
}

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://app:app@postgres:5432/fraud")

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


def show_results():
    """Раздел «Посмотреть результаты»: данные из Postgres"""
    # 1. Последние 10 транзакций с fraud_flag = 1
    st.markdown("**10 последних транзакций с fraud_flag = 1**")
    fraud = query_db(
        "SELECT transaction_id, score, fraud_flag, created_at FROM scores "
        "WHERE fraud_flag = 1 ORDER BY created_at DESC LIMIT 10"
    )
    if fraud.empty:
        st.info("Транзакций с fraud_flag = 1 пока нет")
    else:
        st.dataframe(fraud, use_container_width=True)

    # 2. Гистограмма скоров
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
mode = st.radio("Гистограмма скоров по:", [MODE_LAST_N, MODE_LAST_BATCH])

if st.button("Посмотреть результаты"):
    st.session_state.show_results = True

if st.session_state.get("show_results"):
    try:
        show_results()
    except Exception as e:
        st.error(f"Не удалось получить данные из Postgres: {str(e)}")
