"""Implementation of tests for survey"""

# pylint: disable=redefined-outer-name
from itertools import product, combinations

import pytest
import numpy as np

from seismicpro import Survey, StackingVelocity
from seismicpro.src.utils import to_list
from seismicpro.src.const import HDR_FIRST_BREAK


# Constants
ALL_ATTRS = ['data', 'headers', 'samples', 'sort_by', 'survey']
COPY_IGNORE_ATTRS = ['data', 'headers', 'samples'] # Attrs that might not be copied during gather.copy
NUMPY_ATTRS = ['data', 'samples']


@pytest.fixture(scope='module')
def survey(segy_path):
    """Create gather"""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'],
                    header_cols=['offset', 'FieldRecord'])
    survey.remove_dead_traces(bar=False)
    survey.collect_stats(bar=False)
    survey.headers[HDR_FIRST_BREAK] = np.random.randint(0, 1000, len(survey.headers))
    return survey


@pytest.fixture(scope='function')
def gather(survey):
    """gather"""
    return survey.get_gather((0, 0))


def compare_gathers(first, second, drop_cols=None, check_types=False, same_survey=True):
    """compare_gathers"""
    first_attrs = first.__dict__
    second_attrs = second.__dict__

    assert len(set(first_attrs) & set(second_attrs)) == len(first_attrs)

    first_headers = first.headers.reset_index()
    second_headers = second.headers.reset_index()
    if drop_cols:
        first.validate(required_header_cols=drop_cols)
        second.validate(required_header_cols=drop_cols)

        first_headers.drop(columns=drop_cols, inplace=True)
        second_headers.drop(columns=drop_cols, inplace=True)

    assert len(first_headers) == len(second_headers)
    if len(first_headers) > 0:
        assert first_headers.equals(second_headers)

    assert np.allclose(first.data, second.data)
    assert np.allclose(first.samples, second.samples)

    if check_types:
        for attr in NUMPY_ATTRS:
            first_item = getattr(first, attr)
            second_item = getattr(second, attr)
            if first_item is None or second_item is None:
                assert first_item is second_item
            else:
                assert first_item.dtype.type == second_item.dtype.type

        assert np.all(first.headers.dtypes == second.headers.dtypes)
        assert isinstance(first.sort_by, type(second.sort_by))

    assert first.sort_by == second.sort_by
    if same_survey:
        assert id(first.survey) == id(second.survey)


def test_gather_attrs(gather):
    """This test insist to recheck all tests if new attribute is added."""
    attrs = gather.__dict__
    for name, value in attrs.items():
        assert name in ALL_ATTRS, f"Missing attribute {name} in gather tests."
        if name in NUMPY_ATTRS:
            assert type(value).__module__ == np.__name__, f"The {name} attribute is not related to numpy object and "\
                                                           "must be deleted from NUMPY_ATTRS"
        else:
            assert type(value).__module__ != np.__name__, f"Missing attribute {name} in numpy related tests in gather"



@pytest.mark.parametrize('key', ('offset', ['offset', 'FieldRecord']))
def test_gather_getitem_headers(gather, key):
    """test_gather_getitem_headers"""
    result_getitem = gather[key]
    result_get_item = gather.get_item(key)

    expected = gather.headers[key].values
    assert np.allclose(result_getitem, result_get_item)
    assert result_getitem.shape == (gather.shape[0], len(to_list(key)))
    assert np.allclose(result_getitem.reshape(-1), expected.reshape(-1))
    assert result_getitem.dtype == expected.dtype


simple_keys = [-1, 0, slice(None, 5, None), slice(None, -5, None)]
array_keys = [(0, ), (0, 1), [0], [0, 4], [-1, 5]]

fail_keys = list(product([(0, ), (0, 1), [0], [0, 4], [-1, 5]], repeat=2))
valid_keys = (simple_keys + array_keys + list(product(simple_keys, repeat=2))
              + list(product(simple_keys, array_keys)) + list(product(array_keys, simple_keys)))

@pytest.mark.parametrize('key', valid_keys)
def test_gather_getitem_gathers(gather, key):
    """test_gather_getitem_gathers"""
    result_getitem = gather[key]
    result_get_item = gather.get_item(key)
    expected_data = gather.data[key]

    compare_gathers(result_getitem, result_get_item, check_types=True)
    assert np.allclose(result_getitem.data.reshape(-1), expected_data.reshape(-1))
    assert result_getitem.sort_by == result_get_item.sort_by == gather.sort_by

    # Find a correct shape of data when numpy indexing works differently
    keys = (key, ) if not isinstance(key, tuple) else key
    if result_getitem.shape != expected_data.shape:
        expected_shape = ()
        for k, orig_shape in zip(keys, gather.shape):
            if isinstance(k, int):
                shape_comp = 1
            elif isinstance(k, slice):
                shape_comp = k.stop
                shape_comp = orig_shape + shape_comp if shape_comp < 0 else shape_comp
            else:
                shape_comp = len(k)
            expected_shape = expected_shape + (shape_comp, )

        if len(expected_shape) < 2:
            expected_shape = expected_shape + (gather.shape[1], )
        assert result_getitem.shape == expected_shape, f"for the key {key} expected {result_getitem.shape} shape but "\
                                                       f"received {expected_shape}"

    assert result_getitem.data.shape[0] == len(result_getitem.headers)
    assert result_getitem.data.shape[1] == len(result_getitem.samples)

    # Check that the headers and samples contain  proper values
    ## This is probably not the best way for the equality check..
    keys = tuple(to_list(k) if not isinstance(k, slice) else k for k in keys)
    keys = (keys[0], slice(None)) if len(keys) < 2 else keys
    assert result_getitem.headers.equals(gather.headers.iloc[keys[0]])
    assert np.allclose(result_getitem.samples, gather.samples[keys[1]])


@pytest.mark.parametrize('key', fail_keys)
def test_gather_getitem_gather_fail(gather, key):
    """test_gather_getitem_gathers"""
    pytest.raises(ValueError, gather.__getitem__, key)
    pytest.raises(ValueError, gather.get_item, key)


@pytest.mark.parametrize('key', [[0, 3, 1]])
def test_gather_getitem_sort_by(gather, key):
    """test_gather_getitem_sort_by"""
    result_getitem = gather[key]
    assert result_getitem.sort_by is None


@pytest.mark.parametrize('key, sample_rate', [(slice(None), 2),
                                              (slice(0, 8, 2), 4),
                                              ([1, 2, 3], 2),
                                              ([1, 3, 5], 4),
                                              ([1, 2, 5], None),
                                              (0, None),
                                              ])
def test_gather_getitem_sample_rate_changes(gather, key, sample_rate):
    """test_gather_getitem_sample_rate_changes"""
    result_getitem = gather[slice(None), key]
    if sample_rate is not None:
        assert result_getitem.sample_rate == sample_rate  # pylint: disable=protected-access
    else:
        with pytest.raises(ValueError):
            _ = result_getitem.sample_rate
    if sample_rate is not None:
        assert result_getitem.sample_rate == sample_rate
    else:
        with pytest.raises(ValueError):
            sample_rate = result_getitem.sample_rate


ignore =  [None] + COPY_IGNORE_ATTRS + sum([list(combinations(COPY_IGNORE_ATTRS, r=i)) for i in range(1, 4)], [])
@pytest.mark.parametrize('ignore', ignore)
def test_gather_copy(gather, ignore):
    """test_gather_copy"""
    copy_gather = gather.copy(ignore=ignore)
    ignore = [] if ignore is None else ignore
    for attr in copy_gather.__dict__:
        copy_id = id(getattr(copy_gather, attr))
        orig_id = id(getattr(gather, attr))
        if attr in COPY_IGNORE_ATTRS and attr not in ignore:
            assert copy_id != orig_id
        else:
            assert copy_id == orig_id

    compare_gathers(copy_gather, gather, check_types=True)


@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
@pytest.mark.parametrize('q', [0.1, [0.1, 0.2], (0.1, 0.2), np.array([0.1, 0.2])])
def test_gather_get_quantile(gather, tracewise, use_global, q):
    """Test gahter's methods"""
    # # check that quantile has the same type as q
    gather.get_quantile(q=q, tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_standard(gather, tracewise, use_global):
    """test_gather_scale_standard"""
    gather.scale_standard(tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_minmax(gather, tracewise, use_global):
    """test_gather_scale_minmax"""
    gather.scale_minmax(tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise, use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_maxabs(gather, tracewise, use_global):
    """test_gather_scale_minmax"""
    gather.scale_maxabs(tracewise=tracewise, use_global=use_global)

def test_gather_mask_to_pick_and_pick_to_mask(gather):
    """test_gather_mask_to_pick"""
    mask = gather.pick_to_mask(first_breaks_col=HDR_FIRST_BREAK)
    mask.mask_to_pick(first_breaks_col=HDR_FIRST_BREAK, save_to=gather)

def test_gather_get_coords(gather):
    """test_gather_get_coords"""
    gather.get_coords()


def test_gather_sort(gather):
    """test_gather_sort"""
    gather.sort(by='offset')

def test_gather_validate(gather):
    """test_gather_validate"""
    gather.sort(by='offset')
    gather.validate(required_header_cols=['offset', 'FieldRecord'], required_sorting='offset')

def test_gather_muting(gather):
    """test_gather_muting"""
    offsets = [1000, 2000, 3000]
    times = [100, 300, 600]
    muter = gather.create_muter(mode='points', offsets=offsets, times=times)
    gather.mute(muter)

def test_gather_semblance(gather):
    """test_gather_semblance"""
    gather.sort(by='offset')
    velocities = np.linspace(1300, 5500, 140)
    gather.calculate_semblance(velocities=velocities)

def test_gather_res_semblance(gather):
    """test_gather_res_semblance"""
    gather.sort(by='offset')
    stacking_velocity = StackingVelocity.from_points(times=[0, 3000], velocities=[1600, 3500])
    gather.calculate_residual_semblance(stacking_velocity=stacking_velocity)

def test_gather_stacking_velocity(gather):
    """test_gather_stacking_velocity"""
    gather.sort(by='offset')
    stacking_velocity = StackingVelocity.from_points(times=[0, 3000], velocities=[1600, 3500])
    gather.apply_nmo(stacking_velocity=stacking_velocity)

def test_gather_get_central_cdp(segy_path):
    """test_gather_get_central_cdp"""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'], header_cols=['offset', 'FieldRecord'])
    survey = survey.generate_supergathers()
    gather = survey.get_gather((0, 0))
    gather.get_central_cdp()

def test_gather_stack(gather):
    """test_gather_stack"""
    gather.stack()
