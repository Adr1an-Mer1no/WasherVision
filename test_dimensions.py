from dimension_check import dimension_recommendation

def test_known_w1():
    status, candidate = dimension_recommendation(16, 8, 'W3')
    assert status == 'REVIEW_SIZE_MISMATCH'
    assert candidate == 'W1'

def test_known_w2():
    assert dimension_recommendation(18,11,'W2')[0]=='SIZE_CONSISTENT'

def test_n1_w3_ambiguity():
    assert dimension_recommendation(20,10,'N1')[0]=='SIZE_CONSISTENT'

def test_missing_measurement():
    assert dimension_recommendation(None,None,'W1')[0]=='NO_RELIABLE_MEASUREMENT'
