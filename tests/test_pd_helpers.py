# -*- coding: utf-8 -*-
import pandas as pd

from postpanda_helper.pd_helpers import strip_all_spaces


def test_strip_all_spaces():
    df = pd.DataFrame([["abc", " abc", "def "], ["ab cd", "ab  cd", " ab  cd  "]])
    dfl = strip_all_spaces(df).to_records(index=False).tolist()
    assert dfl == [("abc", "abc", "def"), ("ab cd", "ab cd", "ab cd")]
