import os
import logging
import pickle
import pkg_resources
from typing import Tuple, List

import pandas as pd
import yfinance as yf

log = logging.getLogger("data")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class DataError(Exception):
    pass


def load_symbol(symbol, max_retry, **kwargs) -> Tuple[dict, pd.DataFrame]:
    for i in range(max_retry + 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(**kwargs)
            log.info(f"Loaded {symbol} data with {kwargs}")
            return (info, hist)
        except Exception as e:
            log.warning(f"Failed to get {symbol} data on {i} attempt - {e}")

    raise DataError(f"Cannot get {symbol} data after {max_retry} attempt - {e}")


def write_universe(symbols: List[str], overwrite: bool = True, max_retry: int = 3, **kwargs):
    for symbol in symbols:
        dir = pkg_resources.resource_filename("pair_trading", f"data\\stocks\\{symbol}")

        if not overwrite and os.path.isdir(dir):
            continue  # skip, the data exists

        try:
            info, hist = load_symbol(symbol, max_retry, **kwargs)
            write_symbol(dir, info, hist)
        except Exception:
            continue


def write_symbol(dir, info, hist):
    if not os.path.isdir(dir):
        os.mkdir(dir)

    hist.to_csv(os.path.join(dir, "hist.csv"))
    with open(os.path.join(dir, "info.pkl"), "wb") as f:
        pickle.dump(info, f)

    log.info(f"Written data to {dir}")


def write_info(universe: List[str]):
    info = []
    dir = pkg_resources.resource_filename("pair_trading", "data")
    for stock in universe:
        try:
            with open(os.path.join(dir, f"stocks\\{stock}\\info.pkl"), "rb") as f:
                info.append(pickle.load(f))
        except Exception as e:
            log.warning(f"Failed to get {stock} info - {e}")

    info = pd.DataFrame(info)
    path = os.path.join(dir, "info.csv")
    log.info(f"Writing {path}")
    info.to_csv(path)


def write_items(universe: List[str], items: List[str] = None):
    if items is None:
        items = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]

    hist = {}
    dir = pkg_resources.resource_filename("pair_trading", "data")
    for stock in universe:
        try:
            hist[stock] = pd.read_csv(os.path.join(dir, f"stocks\\{stock}\\hist.csv"), index_col="Date")
        except Exception as e:
            log.warning(f"Failed to get {stock} history - {e}")

    for item in items:
        df = pd.DataFrame({k: v[item] for k, v in hist.items() if item in v}).reindex(columns=universe)
        path = os.path.join(dir, f"{item}.csv")
        log.info(f"Writing {path}")
        df.to_csv(path)


if __name__ == "__main__":
    from pair_trading.universe import UNIVERSE
    write_universe(UNIVERSE, overwrite=False, start="2000-01-01", end="2021-01-01")
    write_info(UNIVERSE)
    write_items(UNIVERSE)
