from collections import deque
from class_fusion import recommend, vote

def test_w1_correction():
    r = recommend('W3', .91, 16.3, 7.9)
    assert r['suggested'] == 'W1'
    assert r['status'] == 'SIZE_SUGGESTS_CORRECTION'

def test_w2_suggestion():
    r = recommend('W3', .95, 18.9, 9.8)
    assert r['status'] in ('AMBIGUOUS_SIZE', 'SIZE_SUGGESTS_CORRECTION')

def test_same_size_ambiguity():
    r = recommend('W1', .9, 20, 10)
    assert r['suggested'] is None
    assert r['status'] == 'AMBIGUOUS_N1_W3'

def test_overlap_abstains():
    assert recommend('W4', .9, 22, 7, overlap=True)['suggested'] is None

def test_missing_id_abstains():
    assert recommend('W4', .9, 22, None)['suggested'] is None

def test_stable_vote():
    assert vote(deque(['W1']*27+[None]*3))[0] == 'W1'

def test_unstable_vote():
    assert vote(deque(['W1']*16+['W2']*14))[0] is None

def test_w2_latest_measurement_provisional():
    r = recommend('W3', .84, 18.8, 10.2)
    assert r['suggested'] == 'W2'
    assert r['status'] == 'PROVISIONAL_W2_SIZE_ADVANTAGE'

def test_w2_tie_abstains():
    assert recommend('W3', .84, 19.0, 10.5)['suggested'] is None

def test_w2_invalid_hole_abstains():
    assert recommend('W3', .84, 18.8, None)['suggested'] is None
