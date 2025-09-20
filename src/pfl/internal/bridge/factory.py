from pfl.internal.bridge.base import (
    CommonFrameworkBridge,
    FedProxFrameworkBridge,
    FTRLFrameworkBridge,
    SCAFFOLDFrameworkBridge,
    SGDFrameworkBridge,
)
from pfl.internal.ops.framework_types import MLFramework
from pfl.internal.ops.selector import get_framework_module

from pfl.internal.bridge.factory import FrameworkBridgeFactory as PFLFrameworkBridgeFactory

class FrameworkBridgeFactory(PFLFrameworkBridgeFactory):
    
    #
    # overwrite sgd bridge methode with custom sgd pytorch bridge
    #
    @staticmethod
    def sgd_bridge() -> SGDFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import sgd as sgd_pt
            return sgd_pt.PyTorchSGDBridge
        elif framework == MLFramework.TENSORFLOW:
            from .tensorflow import sgd as sgd_tf
            return sgd_tf.TFSGDBridge
        elif framework == MLFramework.MLX:
            from .mlx import sgd as sgd_mlx
            return sgd_mlx.MLXSGDBridge
        else:
            raise NotImplementedError("SGD bridge not available "
                                      f"for framework {framework}")
    
    @staticmethod
    def fedprox_bridge() -> FedProxFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import proximal as proximal_pt
            return proximal_pt.PyTorchFedProxBridge
        elif framework == MLFramework.TENSORFLOW:
            from .tensorflow import proximal as proximal_tf
            return proximal_tf.TFFedProxBridge
        else:
            raise NotImplementedError("FedProx bridge not available "
                                      f"for framework {framework}")
    
    @staticmethod
    def scaffold_bridge() -> SCAFFOLDFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import scaffold as scaffold_pt
            return scaffold_pt.PyTorchSCAFFOLDBridge
        else:
            raise NotImplementedError("SCAFFOLD bridge not available "
                                      f"for framework {framework}")

