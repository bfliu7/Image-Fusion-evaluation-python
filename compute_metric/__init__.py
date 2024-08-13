from compute_metric.avg_gradient import __avg_gradient__
from compute_metric.cross_entropy import __cross_entropy__
from compute_metric.edge_intensity import __edge_intensity__
from compute_metric.entropy import __entropy__
from compute_metric.mutinf import __mutinf__
from compute_metric.psnr import __psnr__
from compute_metric.Qabf import __Qabf__
from compute_metric.Qcb import __Qcb__
from compute_metric.Qcv import __Qcv__

__metrics__ = {
    "Avg_gradient":__avg_gradient__,
    "Cross_entropy":__cross_entropy__,
    "Edge_intensity":__edge_intensity__,
    "Entropy":__entropy__,
    "Mutinf":__mutinf__,
    "Psnr": __psnr__,
    "Qabf":__Qabf__,
    "Qcb":__Qcb__,
    # "Qcv":__Qcv__,



}
