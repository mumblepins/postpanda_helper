# coding=utf-8
from os import PathLike
from typing import Union

import yaml
from pandas import DataFrame

from postpanda_helper import SelectSert
from postpanda_helper.pd_helpers import to_string_tuples_drop_val


def handle_exec(obj):
    g = {}
    l = {}
    ret = {}
    exec(obj["code"], g, l)
    for k in obj["keys"]:
        ret[k] = g.get(k, l.get(k))
    return ret


def walk_list(obj):
    for n in range(len(obj)):
        if isinstance(obj[n], list):
            obj[n] = walk_list(obj[n])
        elif isinstance(obj[n], dict):
            obj[n] = walk_dict(obj[n])
    return obj


def walk_dict(obj):
    to_add = {}
    for k in list(obj.keys()):
        if k == "to_exec":
            exec_dict = handle_exec(obj[k])
            del obj[k]
            to_add.update(exec_dict)
        elif isinstance(obj[k], list):
            obj[k] = walk_list(obj[k])
        elif isinstance(obj[k], dict):
            obj[k] = walk_dict(obj[k])

    obj.update(to_add)
    return obj


def walk(obj):
    if isinstance(obj, list):
        return walk_list(obj)
    elif isinstance(obj, dict):
        return walk_dict(obj)
    else:
        return obj


def load_spec(file_path):
    with open(file_path) as fh:
        yd = yaml.safe_load(fh)
    return walk(yd)


def normalizer(
    frame: DataFrame, spec_file: Union[str, PathLike], substituter: SelectSert
) -> list[DataFrame]:
    """Normalizes dataframe based on a YAML spec file

    Modifies dataframe inplace

    **YAML Example:**

    .. code-block:: yaml

        - action:       replace_id
          columns:
            - datasource_id
            - units
          delete_old:   false
          new_col_name: unit_id
          sql_column_names:
            - datasource_id
            - name
          table_name:   units

        - action:           replace_id
          columns:          f
          delete_old:       false
          new_col_name:     frequency_id
          sql_column_names: name
          table_name:       frequency

        - action:  apply_funcs
          to_exec:
            keys:
              - functions
            # language=Python
            code: |
                   from functools import partial
                   import pandas as pd
                   from postpanda_helper.pd_helpers import to_date


                   def col_to_pt(s: pd.Series):
                       import geopandas as gpd

                       latlon = s.str.split(",", expand=True)
                       out = gpd.GeoSeries(gpd.points_from_xy(latlon[1], latlon[0], crs="WGS84"))
                       out[s.isna()] = None
                       return out


                   functions = {
                       None: [
                           partial(to_date, freq_col="f", date_col="end", start_or_end="end"),
                           partial(to_date, freq_col="f", date_col="start", start_or_end="start"),
                       ],
                       "last_updated": lambda x: pd.to_datetime(x, utc=True),
                       "latlon": col_to_pt,
                       "latlon2": col_to_pt,
                   }

        - action: drop_cols
          to_drop:
            - units
            - iso3166
            - unitsshort
            - lon
            - lat
            - lon2
            - lat2
        - action: drop_na_cols

        - action: rename_cols
          name_map:
            end:   end_date
            start: start_date

        - action:      extra_to_json
          excluded:
            - f
            - series_id
            - name
            - description
            - start_date
            - end_date
            - last_updated
            - geoset_id
            - data
            - datasource_id
            - frequency_id
            - source_id
            - unit_id
            - geography_id
            - geography2_id
            - latlon
            - latlon2
          json_column: extra_data

        - action: many_to_many
          columns:
            - tag
            - geography_id
          table:  x_data_points_geography

        - action: combine_to_tuples
          delete_old: true
          columns:
            charge_dep_gal_100mile:
              - city_cd
              - highway_cd
              - combined_cd
            elec_comp_kwh_100mile:
              - city_e
              - highway_e
              - comb_e

    Args:
        frame: frame to modify
        spec_file: path to YAML file
        substituter:
    """
    replace_spec = load_spec(spec_file)
    output = []
    for rs in replace_spec:
        action = rs.pop("action")
        print(action)
        if action == "replace_id":
            substituter.replace_with_ids(frame, **rs)
        elif action == "drop_cols":
            frame.drop(columns=rs["to_drop"], inplace=True, errors="ignore")
        elif action == "apply_funcs":
            for c, f in rs["functions"].items():
                if c is not None and c not in frame:
                    continue
                if not isinstance(f, (tuple, list)):
                    f = [f]
                for sf in f:
                    if c is not None:
                        frame.loc[:, c] = sf(frame.loc[:, c])
                    else:
                        sf(frame)
                # if c in frame:
                #     frame[c] = f(frame[c])
        elif action == "rename_cols":
            frame.rename(columns=rs["name_map"], inplace=True)
        elif action == "extra_to_json":
            cols_for_json = list(set(frame.columns).difference(rs["excluded"]))
            if len(cols_for_json) == 0:
                continue
            frame[rs["json_column"]] = frame[cols_for_json].apply(
                lambda x: x.dropna().to_dict() or None, axis=1
            )
        elif action == "drop_na_cols":
            frame.dropna(axis=1, how="all", inplace=True)
        elif action == "many_to_many":
            substituter.many_many_link(frame, **rs)
        elif action == "combine_to_tuples":
            drop_cols = rs.get("delete_old", False)
            for k, v in rs["columns"].items():
                frame[k] = to_string_tuples_drop_val(frame.reindex(columns=v))
                if drop_cols:
                    frame.drop(columns=set(v) - rs["columns"].keys(), inplace=True, errors="ignore")
        elif action == "output_frame":
            if "columns" in rs:
                output.append(frame.reindex(columns=rs["columns"]).copy())
            else:
                output.append(frame.copy())
        else:
            raise NotImplementedError(f"unknown action: {action}")
    return output
