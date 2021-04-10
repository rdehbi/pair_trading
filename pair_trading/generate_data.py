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
        except DataError:
            continue
        try:
            write_symbol(dir, info, hist)
        except:
            pass


def write_symbol(dir, info, hist):
    if not os.path.isdir(dir):
        os.mkdir(dir)

    hist.to_csv(os.path.join(dir, "hist.csv"))
    with open(os.path.join(dir, "info.pkl"), "wb") as f:
        pickle.dump(info, f)

    log.info(f"Written data to {dir}")


if __name__ == "__main__":
    from pair_trading.universe import UNIVERSE
    write_universe(UNIVERSE, overwrite=False, start="2000-01-01", end="2021-01-01")
