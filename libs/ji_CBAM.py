# 这里是我们加入的CBAM模块部分
# 该模块用于对提取出来的volume feature进行注意力机制
from .cbam import *

def CBAM_module(z_vol):
    """

    Args:
        z_vol:  (float tensor, (bs, c, d, h, w)): feature volumes.

    Returns:
        transformed_z_vol: (float tensor, (bs, c, d, h, w))经过注意力之后的feature volumes

    """
        # 这里现在的维数有点问题，需要改一下
    # raise("dimension error")
    # 这里的参数我们自己添加了一些新的东西
    CBAM_block = CBAM( bs=z_vol.shape[0], c=z_vol.shape[1], gate_channels=z_vol.shape[2], reduction_ratio=4 )
    if torch.cuda.is_available():
        CBAM_block.cuda()  # 将程序移动到GPU上进行并行计算
    transformed_z_vol = CBAM_block(z_vol)

    #检查是否具有相同的shape
    assert transformed_z_vol.shape == z_vol.shape

    return transformed_z_vol

