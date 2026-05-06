# binary

## Overview

**Binary and set-style** distances between aligned vectors: Hamming (mismatch rate) and Jaccard-style distance from boolean masks. [`metric.py`](metric.py) defines `BinaryDistanceMetric`; implementations live under [`calculators/`](calculators/).

## Components

| Path | Description |
|------|-------------|
| [`metric.py`](metric.py) | `BinaryDistanceMetric` enum mapping metric names to calculator instances. |
| [`calculators/`](calculators/) | `HammingDistanceCalculator` and `JaccardDistanceCalculator`. |

## Examples

### Hamming (pairwise)

Normalized mismatch fraction over feature axes; inputs may be any numeric dtype.

```python
import numpy as np
from distance_metric import BinaryDistanceMetric, DistanceCalculator, DistanceResult

calc: DistanceCalculator = BinaryDistanceMetric.HAMMING.calculator
bits: np.ndarray = np.array(
    [[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
    dtype=np.float64,
)
result: DistanceResult = calc.pairwise(array=bits)
assert result.value.shape == (3, 3)
```

### Jaccard (cross)

Nonzero entries are treated as set membership after boolean conversion.

```python
import numpy as np
from distance_metric import BinaryDistanceMetric

calc = BinaryDistanceMetric.JACCARD.calculator
query_array = np.array([[1, 0, 1, 0]], dtype=np.float64)
gallery_array = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float64)
result = calc.cross(query_array=query_array, gallery_array=gallery_array)
assert result.value.shape == (1, 2)
```
