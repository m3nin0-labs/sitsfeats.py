# Metrics catalogue

Every metric reduces a time series $x = (x_1, \dots, x_N)$ of $N$ observations to
a single value. The live list is always available via [`metrics()`](api.md#sitsfeatsmetrics). This page presents the formulation of each metric implemented in the package.

!!! note "Conventions"
    - **Standard deviation** is the *sample* deviation (`ddof = 1`).
    - **Skewness** is the adjusted Fisher–Pearson coefficient
      (`scipy.stats.skew(..., bias=False)`).
    - **Kurtosis** is Pearson's (non-excess) definition: $3.0$ for a normal
      distribution.
    - **Quantiles** use the **Hazen** plotting position
      (`numpy.quantile(..., method="hazen")`).

## Basic family

Let $\bar{x}$ be the mean and $m_k = \tfrac{1}{N}\sum_i (x_i-\bar{x})^k$ the
$k$-th central moment.

| Name | Description | Definition |
|------|-------------|------------|
| `max` | Maximum | $\max_i x_i$ |
| `min` | Minimum | $\min_i x_i$ |
| `mean` | Arithmetic mean | $\bar{x} = \tfrac{1}{N}\sum_i x_i$ |
| `median` | Median | middle value (Hazen $Q_2$) |
| `sum` | Sum | $\sum_i x_i$ |
| `std` | Standard deviation | $\sqrt{\tfrac{1}{N-1}\sum_i (x_i-\bar{x})^2}$ |
| `skew` | Skewness | $\dfrac{\sqrt{N(N-1)}}{N-2}\cdot\dfrac{m_3}{m_2^{3/2}}$ |
| `kurt` | Kurtosis (Pearson) | $\dfrac{N\sum_i (x_i-\bar{x})^4}{\left(\sum_i (x_i-\bar{x})^2\right)^2}$ |
| `amplitude` | Range | $\max_i x_i - \min_i x_i$ |
| `fslope` | Max. abs. first difference | $\max_i \lvert x_{i+1}-x_i \rvert$ |
| `abs_sum` | Absolute sum | $\sum_i \lvert x_i \rvert$ |
| `amd` | Mean abs. first difference | $\tfrac{1}{N-1}\sum_i \lvert x_{i+1}-x_i \rvert$ |
| `mse` | Mean power spectrum | $\tfrac{1}{N}\sum_k \lvert \hat{x}_k \rvert^2 = \sum_i x_i^2$ |
| `fqr` | First quartile | $Q_1$ (Hazen) |
| `tqr` | Third quartile | $Q_3$ (Hazen) |
| `iqr` | Interquartile range | $Q_3 - Q_1$ |

!!! info "`mse`"
    By Parseval's theorem, the mean of $\lvert \text{FFT} \rvert^2$ over all
    frequency bins equals the sum of squared samples, so `mse` is computed
    directly as $\sum_i x_i^2$, with no FFT.

## Polar family

The polar metrics follow the Körting *polar plot* representation used by [stmetrics](https://github.com/brazil-data-cube/stmetrics), and are verified against its Shapely-based implementation.

**Construction.** Each observation is placed on a circle: vertex $i$ at angle $\theta_i = \tfrac{2\pi i}{N}$ with radius $r_i = \lvert x_i \rvert$, i.e. the Cartesian point $(r_i\cos\theta_i,\; r_i\sin\theta_i)$. Connecting consecutive vertices yields a closed polygon $P$. Because radii are non-negative and the vertices are ordered by angle, $P$ is star-shaped about the origin and therefore simple (non-self-intersecting).

| Name | Description | Definition |
|------|-------------|------------|
| `area_ts` | Polygon area | $\tfrac{1}{2}\sin\!\left(\tfrac{2\pi}{N}\right)\sum_i r_i\, r_{i+1}$ |
| `angle` | Phase of the maximum | $\theta_{\arg\max_i r_i}$, with $\theta$ from $\operatorname{linspace}(0, 2\pi, N)$ |
| `gyration_radius` | Spread of the shape | mean distance of $P$'s vertices to its area-centroid |
| `csi` | Cell shape index | $\dfrac{\text{perimeter}(P)^2}{4\pi\,\text{area}(P)}$ |
| `area_q1`…`area_q4` | Seasonal quadrant areas | area of $P$ within each Cartesian quadrant |
| `polar_balance` | Seasonal balance | standard deviation of the four quadrant areas |

The quadrants partition the plane at the origin (`area_q1` = upper-right, `area_q2` = upper-left, `area_q3` = lower-left, `area_q4` = lower-right) and act as the four "seasons" of the cycle.

!!! note "Not yet implemented"
    `ecc_metric` (eccentricity, via the minimum rotated rectangle) from
    stmetrics is not yet available.
