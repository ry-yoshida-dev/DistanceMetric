# distance_metric

## Overview

Top-level package for **pairwise**, **cross**, and **elementwise** distance and similarity computation on NumPy arrays. Public API exposes `DistanceCalculator`, result typing (`DistanceResult`, `DistanceResultType`), the abstract `DistanceMetric` registry pattern, and family-specific enums (for example `MinkowskiDistanceMetric`) that resolve to concrete calculators under [`calculators/`](calculators/).

## Components

| Path | Description |
|------|-------------|
| [`calculator.py`](calculator.py) | Abstract `DistanceCalculator`: `pairwise`, `cross`, `elementwise`, and `_cross_array` hook for subclasses. |
| [`metric.py`](metric.py) | `DistanceMetric` abstract enum base with a `.calculator` property for each registry member. |
| [`result.py`](result.py) | `DistanceResult` dataclass and `DistanceResultType` (`DISTANCE` vs `SIMILARITY`) with `best_score` / `worst_score` helpers. |
| [`calculators/`](calculators/) | Metric families, enums, and concrete calculator implementations. |

## Examples

Install the package from the repository root (for example `pip install -e .`). When running snippets without installing, set `PYTHONPATH` to the `src` directory so `import distance_metric` resolves.

### Pairwise matrix via enum

```python
import numpy as np
from distance_metric import DistanceCalculator, DistanceResult, MinkowskiDistanceMetric

calc: DistanceCalculator = MinkowskiDistanceMetric.EUCLIDEAN.calculator
x: np.ndarray = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
result: DistanceResult = calc.pairwise(array=x)
assert result.value.shape == (2, 2)
```

### Cross query vs gallery

```python
import numpy as np
from distance_metric import DistanceCalculator, DistanceResult, MinkowskiDistanceMetric

calc: DistanceCalculator = MinkowskiDistanceMetric.MANHATTAN.calculator
query_array: np.ndarray = np.array([[1.0, 2.0]], dtype=np.float64)
gallery_array: np.ndarray = np.array([[4.0, 6.0], [0.0, 0.0]], dtype=np.float64)
result: DistanceResult = calc.cross(query_array=query_array, gallery_array=gallery_array)
assert result.value.shape == (1, 2)
```

### Elementwise single pair

```python
import numpy as np
from distance_metric import MinkowskiDistanceMetric

calc = MinkowskiDistanceMetric.EUCLIDEAN.calculator
result = calc.elementwise(
    query_array=np.array([1.0, 2.0], dtype=np.float64),
    gallery_array=np.array([4.0, 6.0], dtype=np.float64),
)
assert result.value.shape == (1,)
```

### Working with `DistanceResult`

```python
import numpy as np
from distance_metric import MinkowskiDistanceMetric

result = MinkowskiDistanceMetric.EUCLIDEAN.calculator.pairwise(
    array=np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64),
)
# For DISTANCE results, smaller is better.
best: float = result.best_score
worst: float = result.worst_score
```

Metric-specific kwargs (for example `cov` for Mahalanobis, `radius` for Haversine, `delta` for Huber) are documented in each family README under [`calculators/`](calculators/).
