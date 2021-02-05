import json
import pandas as pd
import numpy as np


def try_convert_to_dict(val):
    if pd.notnull(val):
        try:
            obj = json.loads(val)
            if isinstance(obj, dict):
                return obj
            else:
                raise Exception('Not a json dictionary (could be an int because json.loads is weird)!')
        except Exception:
            return dict(val)
    else:
        return {}


def unnest_df(df):
    original_columns = df.columns
    unnested = 0
    print(len(df))
    for col in original_columns:
        try:
            json_col = df[col].apply(try_convert_to_dict)
            if np.sum(len(x) for x in json_col) == 0:
                raise Exception('Empty column !')
        except:
            continue

        unnested += 1
        unnested_df = pd.json_normalize(json_col)
        unnested_df.columns = [col + '.' + str(subcol) for subcol in unnested_df.columns]
        df = df.drop(columns=[col])
        for unnested_col in unnested_df.columns:
            df[col] = unnested_df[unnested_col]

    return df, unnested
