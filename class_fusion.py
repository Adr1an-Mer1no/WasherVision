"""Conservative OD/ID and YOLO fusion; recommendations are NOT motor commands."""
from collections import Counter, deque
from math import exp
from vision_utils import WASHERS


def recommend(yolo_class, confidence, od, inner_d, overlap=False, measurement_quality='OD_ID_MEASURED', od_sigma=0.8, id_sigma=0.8):
    """Return advisory recommendation and status. Missing or unreliable data abstains.

    Gaussian-like dimensional scores are relative compatibility, NOT calibrated probabilities.
    N1 and W3 are intentionally unresolved by geometry alone.
    """
    if overlap:
        return {'suggested': None, 'status': 'REVIEW_OVERLAP', 'candidates': [], 'scores': {}}
    if od is None or inner_d is None or measurement_quality != 'OD_ID_MEASURED':
        return {'suggested': None, 'status': 'MEASUREMENT_UNAVAILABLE', 'candidates': [], 'scores': {}}
    if od_sigma <= 0 or id_sigma <= 0:
        raise ValueError('Measurement scales must be positive')
    scores = {}
    for name, spec in WASHERS.items():
        do = (od - spec['od_mm']) / od_sigma
        di = (inner_d - spec['id_mm']) / id_sigma
        scores[name] = exp(-0.5 * (do * do + di * di))
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    best_score = scores[best]
    plausible = [name for name in ranked if scores[name] >= best_score * 0.35 and best_score >= 0.10]
    if best_score < 0.10:
        return {'suggested': None, 'status': 'SIZE_OUT_OF_RANGE', 'candidates': [], 'scores': scores}
    # Equal-size classes: geometry cannot distinguish nylon N1 from metal W3.
    if best in ('N1', 'W3'):
        if any(name not in ('N1', 'W3') for name in plausible):
            return {'suggested': None, 'status': 'AMBIGUOUS_SIZE',
                    'candidates': plausible, 'scores': scores}
        if yolo_class in ('N1', 'W3') and confidence >= 0.75:
            return {'suggested': yolo_class, 'status': 'VISUAL_N1_W3_UNVERIFIED',
                    'candidates': ['N1', 'W3'], 'scores': scores}
        return {'suggested': None, 'status': 'AMBIGUOUS_N1_W3',
                'candidates': ['N1', 'W3'], 'scores': scores}
    # Targeted W2 diagnostic: near W2 dimensions, a small but consistent
    # dimensional advantage over W3/N1 can justify a *provisional* W2 vote.
    # This is an engineering heuristic to validate, not a trained probability.
    if best == 'W2' and len(plausible) > 1:
        runner_up = max((scores[k] for k in scores if k != 'W2'), default=0.0)
        if (17.0 <= od <= 19.2 and 9.5 <= inner_d <= 11.8
                and best_score >= 0.25 and best_score >= 1.10 * runner_up):
            return {'suggested': 'W2', 'status': 'PROVISIONAL_W2_SIZE_ADVANTAGE',
                    'candidates': plausible, 'scores': scores}
    if len(plausible) > 1:
        return {'suggested': None, 'status': 'AMBIGUOUS_SIZE', 'candidates': plausible, 'scores': scores}
    return {'suggested': best,
            'status': 'SIZE_AGREES_YOLO' if best == yolo_class else 'SIZE_SUGGESTS_CORRECTION',
            'candidates': [best], 'scores': scores}


def vote(history, window=30, min_votes=20, agreement=0.8):
    """Vote only over accepted per-frame suggestions. Abstentions remain in denominator."""
    recent = list(history)[-window:]
    if len(recent) < window:
        return None, 'COLLECTING', len(recent), 0.0
    valid = [x for x in recent if x is not None]
    if len(valid) < min_votes:
        return None, 'INSUFFICIENT_VALID_FRAMES', len(valid), 0.0
    name, count = Counter(valid).most_common(1)[0]
    fraction = count / len(recent)
    if fraction < agreement:
        return None, 'UNSTABLE_CLASS', count, fraction
    return name, 'STABLE_ADVISORY', count, fraction
