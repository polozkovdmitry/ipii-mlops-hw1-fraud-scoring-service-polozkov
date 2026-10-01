-- Витрина со скорами транзакций. Выполняется один раз при инициализации пустого тома pgdata.
CREATE TABLE IF NOT EXISTS scores (
    transaction_id TEXT PRIMARY KEY,
    score          DOUBLE PRECISION NOT NULL,
    fraud_flag     SMALLINT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS scores_created_idx ON scores (created_at DESC);
