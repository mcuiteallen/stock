# tw-stock-collect-forecast
collect tw stock data daily and forecast price

## postgres docker run
    docker pull postgres
    docker run --name pg-docker -e POSTGRES_PASSWORD=docker -d -p 5432:5432 postgres

## create tabe
    CREATE TABLE stock_history (
        code VARCHAR NOT null,
        type VARCHAR NOT NULL,
        date VARCHAR NOT NULL,
        capacity NUMERIC(20,2) NOT NULL,
        turnover NUMERIC(20,2) NOT NULL,
        open NUMERIC(20,2) NOT NULL,
        high NUMERIC(20,2) NOT NULL,
        low NUMERIC(20,2) NOT NULL,
        close NUMERIC(20,2) NOT NULL,
        rsi NUMERIC(20,2),
        williams NUMERIC(20,2),
        macd NUMERIC(20,2),
        macdsignal NUMERIC(20,2),
        macdhist NUMERIC(20,2),
        upperband NUMERIC(20,2),
        middleband NUMERIC(20,2),
        lowerband NUMERIC(20,2),
        PRIMARY KEY(code, type, date)
    )
    
    CREATE INDEX index_code ON stock_history (code)
    CREATE INDEX index_type ON stock_history (type)
    CREATE INDEX index_date ON stock_history (date)

## how to usage
    python TwHistory.py
