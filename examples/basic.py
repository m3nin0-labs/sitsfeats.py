#
# Copyright (C) 2024 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Example with basic metrics."""

#
# Libraries
#
import numpy as np

from sitsfeats import feats

#
# Load example data
#
data = np.genfromtxt("examples/data.csv", delimiter=",", skip_header=1)

#
# Generate metrics
#
feats(["median", "skew"], data)
# > {'median': array([[5585. ],
# >         [5049. ],
# >         [6601.5],
# >         [6047. ],
# >         [4696. ]]),
# >  'skew': array([[ 0.79073819],
# >         [-0.03741349],
# >         [ 0.34681313],
# >         [ 0.29268245],
# >         [-0.32793692]])}
