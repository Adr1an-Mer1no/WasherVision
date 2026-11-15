"""Fit independent regularized affine corrections to OD and ID using known washer samples."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from vision_utils import WASHERS

FIELDS = ['class_name', 'raw_od_mm', 'raw_id_mm']

def load_samples(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    samples = []
    for r in rows:
        if r.get('class_name') not in WASHERS:
            raise ValueError('Unknown class in samples: ' + str(r.get('class_name')))
        samples.append((r['class_name'], float(r['raw_od_mm']), float(r['raw_id_mm'])))
    return samples

def fit(samples, ridge=3.0):
    """Fit y = a*x + b, shrink slope to 1 and intercept to 0 to avoid overfitting."""
    if len(samples) < 10 or len(set(s[0] for s in samples)) != 5:
        raise ValueError('Collect at least two samples of EACH of the five classes (10+ total).')
    result = {}
    for axis, index, nominal in [('od', 1, 'od_mm'), ('id', 2, 'id_mm')]:
        x = np.array([s[index] for s in samples], dtype=float)
        y = np.array([WASHERS[s[0]][nominal] for s in samples], dtype=float)
        X = np.column_stack([x, np.ones(len(x))])
        prior = np.array([1.0, 0.0])
        reg = np.diag([ridge, ridge])
        a, b = np.linalg.solve(X.T @ X + reg, X.T @ y + reg @ prior)
        if not (0.80 <= a <= 1.20 and abs(b) <= 3.0):
            raise ValueError(f'{axis} correction is too large: a={a:.3f}, b={b:.3f}. Check calibration and samples.')
        result[axis] = {'scale': float(a), 'offset': float(b)}
    return result

def correct(od, inner_d, calibration):
    if od is None or inner_d is None or calibration is None:
        return od, inner_d
    return (od * calibration['od']['scale'] + calibration['od']['offset'],
            inner_d * calibration['id']['scale'] + calibration['id']['offset'])

def evaluate(samples, model):
    raw, adjusted = [], []
    for name, od, inner in samples:
        spec = WASHERS[name]
        co, ci = correct(od, inner, model)
        raw.extend([abs(od-spec['od_mm']), abs(inner-spec['id_mm'])])
        adjusted.extend([abs(co-spec['od_mm']), abs(ci-spec['id_mm'])])
    return float(np.mean(raw)), float(np.mean(adjusted))

def train_and_validate(samples):
    # Hold out the last sample of each class; do not validate on training samples.
    training, heldout = [], []
    for name in WASHERS:
        matching = [s for s in samples if s[0] == name]
        if len(matching) < 3:
            raise ValueError('Need at least 3 independent placements for EACH class (15 samples).')
        training.extend(matching[:-1]); heldout.append(matching[-1])
    candidate = fit(training)
    before, after = evaluate(heldout, candidate)
    if after >= before:
        raise ValueError(f'Holdout error worsened: raw {before:.3f} mm, corrected {after:.3f} mm. Correction NOT saved.')
    model = fit(samples)
    return model, before, after

def main():
    p = argparse.ArgumentParser(description='Fit OD/ID corrections from isolated known washer samples')
    p.add_argument('--samples', default='dimension_samples.csv')
    p.add_argument('--output', default='dimension_calibration.json')
    args = p.parse_args()
    samples = load_samples(args.samples)
    model, before, after = train_and_validate(samples)
    print(f'Holdout mean absolute dimension error: {before:.3f} -> {after:.3f} mm')
    print('Model:', model)
    Path(args.output).write_text(json.dumps({'version': 1, 'od': model['od'], 'id': model['id'],
      'holdout_raw_mae_mm': before, 'holdout_corrected_mae_mm': after}, indent=2))
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
