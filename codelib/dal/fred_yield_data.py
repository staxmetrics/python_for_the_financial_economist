import pandas as pd
import requests
from io import BytesIO


def get_nominal_yield_data(output_type: str = 'zero_yields') -> pd.DataFrame:

    """
    Function that returns nominal yield curve data from https://www.federalreserve.gov/data/nominal-yield-curve.htm

    Parameters
    ----------
    output_type
        Output type: 'parameters', 'zero_yields', 'par_yield', 'inst_forward', 'one_year_forward'

    Returns
    -------
    pd.DataFrame
        DataFrame with observations

    """

    url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses

    yield_data = pd.read_csv(BytesIO(response.content), skiprows=9, index_col=0, parse_dates=True)

    if output_type == "parameters":
        return yield_data[['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']]
    elif output_type == "zero_yields":
        not_zero_yield_cols = [c for c in yield_data.columns if c.lower()[:5] != 'sveny']
        return yield_data.drop(not_zero_yield_cols, axis=1)
    elif output_type == "par_yields":
        not_par_yield_cols = [c for c in yield_data.columns if c.lower()[:6] != 'svenpy']
        return yield_data.drop(not_par_yield_cols, axis=1)
    elif output_type == "inst_forward":
        not_inst_forward_cols = [c for c in yield_data.columns if c.lower()[:5] != 'svenf']
        return yield_data.drop(not_inst_forward_cols, axis=1)
    elif output_type == "one_year_forward":
        not_one_year_forward_cols = [c for c in yield_data.columns if c.lower()[:6] != 'sven1f']
        return yield_data.drop(not_one_year_forward_cols, axis=1)
    else:
        raise ValueError("output_type must be either: 'parameters', 'zero_yields', 'par_yield', 'inst_forward', 'one_year_forward'")


def get_real_yield_data(output_type: str = 'zero_yields') -> pd.DataFrame:

    """
    Function that returns real yield curve data from https://www.federalreserve.gov/data/real-yield-curve.htm

    Parameters
    ----------
    output_type
        Output type: 'parameters', 'zero_yields', 'par_yield', 'zero_beir', 'par_beir'

    Returns
    -------
    pd.DataFrame
        DataFrame with observations

    """

    url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses

    yield_data = pd.read_csv(BytesIO(response.content), skiprows=18, index_col=0, parse_dates=True)

    if output_type == "parameters":
        return yield_data[['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']]
    elif output_type == "zero_yields":
        not_zero_yield_cols = [c for c in yield_data.columns if c.lower()[:5] != 'tipsy'
                               or c.lower()[:6] == 'tipspy']
        return yield_data.drop(not_zero_yield_cols, axis=1)
    elif output_type == "par_yields":
        not_par_yield_cols = [c for c in yield_data.columns if c.lower()[:6] != 'tipspy']
        return yield_data.drop(not_par_yield_cols, axis=1)
    elif output_type == "zero_beir":
        not_zero_beir_cols = [c for c in yield_data.columns if c.lower()[:6] != 'bkeven'
                              or c.lower()[:7] in ['bkevenp', 'bkevenf']
                              or c.lower()[:8] in ['bkeven1f', 'bkevenp', 'bkeven5f']]
        return yield_data.drop(not_zero_beir_cols, axis=1)
    elif output_type == "par_beir":
        not_par_beir_cols = [c for c in yield_data.columns if c.lower()[:7] != 'bkevenp']
        return yield_data.drop(not_par_beir_cols, axis=1)
    else:
        raise ValueError("output_type must be either: 'parameters', 'zero_yields', 'par_yield', 'zero_beir', 'par_beir'")


if __name__ == "__main__":

    df_nominal_yield_data = get_nominal_yield_data(output_type='zero_yields')
    df_real_yields_data = get_real_yield_data(output_type='zero_yields')