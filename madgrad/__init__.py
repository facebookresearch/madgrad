
from .madgrad import MADGRAD

try:
    from .fairseq_madgrad import FairseqMADGRAD
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
