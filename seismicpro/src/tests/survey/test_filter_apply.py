"""Test Survey filter and apply methods"""

import pytest
import pandas as pd

from . import assert_surveys_equal, assert_survey_processed_inplace


class TestFilter:
    @pytest.mark.parametrize("cols", ["offset", ["SourceX", "SourceY"]])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_no_axis(self, survey, cols, inplace):
        survey_copy = survey.copy()
        survey_filtered = survey.filter(lambda df: pd.DataFrame([True] + [False] * (len(df) - 1)), cols=cols,
                                        axis=None, unpack_args=False, inplace=inplace)
        survey_copy.headers = survey_copy.headers.iloc[:1]
        assert_surveys_equal(survey_copy, survey_filtered)
        assert_survey_processed_inplace(survey, survey_filtered, inplace)

    def test_no_axis_unpack(self, survey):
        survey_filtered = survey.filter(lambda sx, sy: sx < sy, cols=["SourceX", "SourceY"], axis=None,
                                        unpack_args=True, inplace=False)
        survey.headers = survey.headers[survey.headers["SourceX"] < survey.headers["SourceY"]]
        assert_surveys_equal(survey, survey_filtered)

    def test_rowwise(self, survey):
        survey_filtered = survey.filter(lambda coords: (coords >= 2).all(), cols=["SourceX", "SourceY"], axis=1,
                                        unpack_args=False, inplace=False)
        survey.headers = survey.headers[(survey.headers["SourceX"] >= 2) & (survey.headers["SourceY"] >= 2)]
        assert_surveys_equal(survey, survey_filtered)

    def test_rowwise_unpack(self, survey):
        survey_filtered = survey.filter(lambda gx, gy: gx < gy, cols=["GroupX", "GroupY"], axis=1, unpack_args=True,
                                        inplace=False)
        survey.headers = survey.headers[survey.headers["GroupX"] < survey.headers["GroupY"]]
        assert_surveys_equal(survey, survey_filtered)

    @pytest.mark.parametrize("axis, unpack_args", [
        [None, False],
        [None, True],
        [0, False],
        [1, False],
    ])
    def test_single_col_equality(self, survey, axis, unpack_args):
        survey_filtered = survey.filter(lambda offset: offset == 0, cols="offset", axis=axis, unpack_args=unpack_args,
                                        inplace=False)
        survey.headers = survey.headers[survey.headers["offset"] == 0]
        assert_surveys_equal(survey, survey_filtered)

    def test_rowwise_equality(self, survey):
        survey_filtered = survey.filter(lambda df: (df < 2).any(axis=1), cols=["INLINE_3D", "CROSSLINE_3D"], axis=None,
                                        unpack_args=False, inplace=False)
        survey_filtered_rows = survey.filter(lambda row: (row < 2).any(), cols=["INLINE_3D", "CROSSLINE_3D"], axis=1,
                                             unpack_args=False, inplace=False)
        assert_surveys_equal(survey_filtered, survey_filtered_rows)

    def test_empty_filter(self, survey):
        survey_filtered = survey.filter(lambda x: False, cols="offset", axis=1, inplace=False)
        survey_copy = survey.copy()
        survey_copy.headers = survey_copy.headers.iloc[0:0]
        assert_surveys_equal(survey_copy, survey_filtered)

    def test_wrong_shape_fail(self, survey):
        with pytest.raises(ValueError):
            survey.filter(lambda df: df, cols=["SourceX", "SourceY"], axis=None, inplace=False)

    def test_wrong_dtype_fail(self, survey):
        with pytest.raises(ValueError):
            survey.filter(lambda offset: offset, cols="offset", axis=None, inplace=False)

    def test_columnwise_fail(self, survey):
        with pytest.raises(ValueError):
            survey.filter(lambda x: x < 0, cols=["GroupX", "GroupY"], axis=0, inplace=False)


class TestApply:
    @pytest.mark.parametrize("cols, res_cols", [
        ["offset", None],
        ["CDP", "CDP_NEW"],
        [["SourceX", "SourceY"], None],
        [["SourceX", "SourceY"], ["GroupX", "GroupY"]],
    ])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_no_axis(self, survey, cols, res_cols, inplace):
        survey_copy = survey.copy()
        survey_applied = survey.apply(lambda df: df + 1, cols=cols, res_cols=res_cols, axis=None, unpack_args=False,
                                      inplace=inplace)
        if res_cols is None:
            res_cols = cols
        survey_copy.headers[res_cols] = survey_copy.headers[cols] + 1
        assert_surveys_equal(survey_copy, survey_applied)
        assert_survey_processed_inplace(survey, survey_applied, inplace)

    def test_no_axis_unpack(self, survey):
        survey_applied = survey.apply(lambda sx, gx: (sx + gx) // 2, cols=["SourceX", "GroupX"], res_cols="TraceX",
                                      axis=None, unpack_args=True, inplace=False)
        survey.headers["TraceX"] = (survey.headers["SourceX"] + survey.headers["GroupX"]) // 2
        assert_surveys_equal(survey, survey_applied)

    def test_columnwise(self, survey):
        survey_applied = survey.apply(lambda coord: coord / coord.max(), cols=["SourceX", "SourceY"],
                                      res_cols=["NormSourceX", "NormSourceY"], axis=0, unpack_args=False,
                                      inplace=False)
        survey.headers["NormSourceX"] = survey.headers["SourceX"] / survey.headers["SourceX"].max()
        survey.headers["NormSourceY"] = survey.headers["SourceY"] / survey.headers["SourceY"].max()
        assert_surveys_equal(survey, survey_applied)

    def test_columnwise_unpack(self, survey):
        survey_applied = survey.apply(lambda *coords: [coords[0]] * len(coords), cols="SourceX", axis=0,
                                      unpack_args=True, inplace=False)
        survey.headers["SourceX"] = survey.headers["SourceX"].iloc[0]
        assert_surveys_equal(survey, survey_applied)

    def test_rowwise(self, survey):
        survey_applied = survey.apply(lambda coords: coords.mean(), cols=["SourceX", "GroupX"], res_cols="TraceX",
                                      axis=1, unpack_args=False, inplace=False)
        survey.headers["TraceX"] = (survey.headers["SourceX"] + survey.headers["GroupX"]) / 2
        assert_surveys_equal(survey, survey_applied)

    def test_rowwise_unpack(self, survey):
        survey_applied = survey.apply(lambda sx, sy, gx, gy: (abs(sx - gx), abs(sy - gy)),
                                      cols=["SourceX", "SourceY", "GroupX", "GroupY"], res_cols=["TraceX", "TraceY"],
                                      axis=1, unpack_args=True, inplace=False)
        survey.headers["TraceX"] = (survey.headers["SourceX"] - survey.headers["GroupX"]).abs()
        survey.headers["TraceY"] = (survey.headers["SourceY"] - survey.headers["GroupY"]).abs()
        assert_surveys_equal(survey, survey_applied, ignore_dtypes=True)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("cols", ["offset", ["CDP_X", "CDP_Y"]])
    def test_independent_processing_equality(self, survey, cols, axis):
        survey_applied = survey.apply(lambda x: x**2, cols=cols, axis=axis, inplace=False)
        survey.headers[cols] **= 2
        assert_surveys_equal(survey, survey_applied)

    @pytest.mark.parametrize("axis, unpack_args", [
        [None, False],
        [None, True],
        [0, False],
        [1, False],
    ])
    def test_single_col_equality(self, survey, axis, unpack_args):
        survey_applied = survey.apply(lambda offset: offset**2, cols="offset", axis=axis, unpack_args=unpack_args,
                                      inplace=False)
        survey.headers["offset"] **= 2
        assert_surveys_equal(survey, survey_applied)

    def test_columnwise_equality(self, survey):
        survey_applied = survey.apply(lambda df: df - df.min(axis=0), cols=["INLINE_3D", "CROSSLINE_3D"], axis=None,
                                      unpack_args=False, inplace=False)
        survey_applied_cols = survey.apply(lambda col: col - col.min(), cols=["INLINE_3D", "CROSSLINE_3D"], axis=0,
                                           unpack_args=False, inplace=False)
        assert_surveys_equal(survey_applied, survey_applied_cols)

    def test_rowwise_equality(self, survey):
        survey_applied = survey.apply(lambda df: df.max(axis=1), cols=["INLINE_3D", "CROSSLINE_3D"],
                                      res_cols="MAX_LINE", axis=None, unpack_args=False, inplace=False)
        survey_applied_rows = survey.apply(lambda row: row.max(), cols=["INLINE_3D", "CROSSLINE_3D"],
                                           res_cols="MAX_LINE", axis=1, unpack_args=False, inplace=False)
        assert_surveys_equal(survey_applied, survey_applied_rows)
