# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .madgrad import MADGRAD
from .mirror_madgrad import MirrorMADGRAD

try:
    from .fairseq_madgrad import FairseqMADGRAD
    from .fairseq_mirror_madgrad import FairSeqMirrorMADGRAD
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
