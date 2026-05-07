import os
import asyncio
import sys
from datetime import datetime

import alpaca_trade_api as tradeapi
import backtrader as bt
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import crud
import models
from database import SessionLocal, engine
from predictor import train_and_predict

sys.path.append("..")
from strategies.SmaCross import SmaCross

try:
    models.Base.metadata.create_all(bind=engine)
except Exception:
    pass

api = tradeapi.REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    "https://paper-api.alpaca.markets",
    api_version="v2",
)

app = FastAPI(title="ML Trading Bot API")

origins = [
    "http://localhost:3000",
    "https://trading-bot-full-stack.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def fetch_market_data(symbol: str, start: str = "2021-01-01") -> pd.DataFrame:
    data_df = api.get_bars(
        symbol,
        "1Day",
        start=start,
        end=datetime.now().strftime("%Y-%m-%d"),
        feed="iex",
    ).df

    if data_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")

    data_df = data_df.copy()

    if isinstance(data_df.index, pd.MultiIndex):
        data_df = data_df.reset_index()
        if "timestamp" in data_df.columns:
            data_df = data_df.set_index("timestamp")
        elif "time" in data_df.columns:
            data_df = data_df.set_index("time")

    required_cols = ["open", "high", "low", "close", "volume"]
    lower_map = {col.lower(): col for col in data_df.columns}
    missing = [col for col in required_cols if col not in lower_map]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing required market data columns: {missing}")

    rename_map = {lower_map[col]: col for col in required_cols}
    data_df = data_df.rename(columns=rename_map)
    data_df = data_df.sort_index()
    data_df["openinterest"] = 0
    data_df["sma_20"] = data_df["close"].rolling(20).mean()
    return data_df


def build_chart_data(data_df: pd.DataFrame) -> list[dict]:
    chart_df = data_df.copy()
    chart_df = chart_df.tail(180)
    chart_df["timestamp"] = pd.to_datetime(chart_df.index).strftime("%Y-%m-%d")
    chart_df = chart_df.fillna(method="bfill").fillna(method="ffill")
    return chart_df[["timestamp", "close", "sma_20", "volume"]].to_dict(orient="records")


def persist_backtest_result(db: Session, symbol: str, starting_value: float, final_value: float) -> None:
    try:
        crud.create_backtest_result(
            db=db,
            symbol_passed=symbol,
            starting_value_passed=starting_value,
            final_value_passed=final_value,
        )
    except Exception:
        db.rollback()


@app.get("/status")
def get_status():
    return {"status": "ok"}


@app.get("/api/account")
def get_account_info():
    try:
        account = api.get_account()
        return {
            "account_number": account.account_number,
            "cash": account.cash,
            "portfolio_value": account.portfolio_value,
            "buying_power": account.buying_power,
            "status": account.status,
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/backtest/{symbol}")
def run_backtest(symbol: str, db: Session = Depends(get_db)):
    try:
        normalized_symbol = symbol.upper().strip()
        data_df = fetch_market_data(normalized_symbol, start="2021-01-01")

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        cerebro.adddata(bt.feeds.PandasData(dataname=data_df))
        cerebro.addstrategy(SmaCross)
        start_val = float(cerebro.broker.getvalue())
        cerebro.run()
        final_val = round(float(cerebro.broker.getvalue()), 2)

        persist_backtest_result(
            db=db,
            symbol=normalized_symbol,
            starting_value=start_val,
            final_value=final_val,
        )

        ml_snapshot = train_and_predict(data_df)

        return {
            "symbol": normalized_symbol,
            "starting_value": round(start_val, 2),
            "final_value": final_val,
            "absolute_return": round(final_val - start_val, 2),
            "percent_return": round(((final_val - start_val) / start_val) * 100, 2),
            "chart_data": build_chart_data(data_df),
            "ml_overlay": {
                "prediction": ml_snapshot["prediction"],
                "confidence": ml_snapshot["confidence"],
                "signal_strength": ml_snapshot["signal_strength"],
                "market_regime": ml_snapshot["market_regime"],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


live_tasks = {}


async def run_live_trade(symbol: str):
    db = SessionLocal()
    try:
        while symbol in live_tasks:
            try:
                latest_bars = fetch_market_data(symbol, start="2024-01-01").tail(120)

                cerebro = bt.Cerebro()
                cerebro.adddata(bt.feeds.PandasData(dataname=latest_bars))
                cerebro.addstrategy(SmaCross)

                results = cerebro.run()
                latest_signal = results[0].crossover[0]
                positions = {position.symbol: position for position in api.list_positions()}

                if symbol not in positions and latest_signal > 0:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=1,
                        side="buy",
                        type="market",
                        time_in_force="day",
                    )
                    try:
                        crud.create_live_trade(
                            db=db,
                            symbol_passed=symbol,
                            side_passed="buy",
                            quantity_passed=1,
                            price_passed=float(order.filled_avg_price or 0),
                        )
                    except Exception:
                        db.rollback()
                elif symbol in positions and latest_signal < 0:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=positions[symbol].qty,
                        side="sell",
                        type="market",
                        time_in_force="day",
                    )
                    try:
                        crud.create_live_trade(
                            db=db,
                            symbol_passed=symbol,
                            side_passed="sell",
                            quantity_passed=float(positions[symbol].qty),
                            price_passed=float(order.filled_avg_price or 0),
                        )
                    except Exception:
                        db.rollback()
            except Exception:
                pass

            await asyncio.sleep(60)
    finally:
        db.close()


@app.post("/api/livetrade/start/{symbol}")
def start_live_trade(symbol: str, background_tasks: BackgroundTasks):
    normalized_symbol = symbol.upper().strip()
    if normalized_symbol in live_tasks:
        raise HTTPException(status_code=400, detail="Live trading for this symbol is already running.")

    live_tasks[normalized_symbol] = True
    background_tasks.add_task(run_live_trade, normalized_symbol)
    return {"message": f"Live trading started for {normalized_symbol}."}


@app.post("/api/livetrade/stop/{symbol}")
def stop_live_trade(symbol: str):
    normalized_symbol = symbol.upper().strip()
    if normalized_symbol not in live_tasks:
        raise HTTPException(status_code=404, detail="Live trading for this symbol is not running.")

    del live_tasks[normalized_symbol]
    return {"message": f"Live trading stopped for {normalized_symbol}."}


def get_prediction_payload(symbol: str) -> dict:
    normalized_symbol = symbol.upper().strip()
    data_df = fetch_market_data(normalized_symbol, start="2020-01-01")
    result = train_and_predict(data_df)
    result["symbol"] = normalized_symbol
    result["as_of"] = datetime.now().strftime("%Y-%m-%d")
    result["chart_data"] = build_chart_data(data_df)
    return result


@app.get("/api/predict/{symbol}")
def predict_stock(symbol: str):
    try:
        return get_prediction_payload(symbol)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ml-insights/{symbol}")
def ml_insights(symbol: str):
    try:
        return get_prediction_payload(symbol)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
