# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .madgrad import MADGRAD

try:
    from .fairseq_madgrad import FairseqMADGRAD
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
