from dimension_calibration import fit, correct, train_and_validate
from vision_utils import WASHERS
import pytest

def test_shared_bias_correction():
    samples=[]
    for name, spec in WASHERS.items():
        for shift in [-0.05, 0, 0.05, 0.02]:
            samples.append((name, spec['od_mm']+0.7+shift, spec['id_mm']-0.6+shift))
    model, before, after = train_and_validate(samples)
    assert after < before
    od, inner = correct(18.7, 10.4, model)
    assert abs(od-18) < .3 and abs(inner-11) < .3

def test_requires_all_classes():
    with pytest.raises(ValueError):
        fit([('W2',18.8,10.0)]*15)
