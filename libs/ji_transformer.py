from .nlost_modules import WindowEncoderSep
import torch

def transformer_module(z_vol):
    """
    Args:
        z_vol:  (float tensor, (bs, c, d, h, w)): feature volumes.

    Returns:
        transformed_z_vol: (float tensor, (bs, c, d, h, w))经过transformer之后的feature volumes

    """
    z_vol_dim = z_vol.shape[1]
    input_resolution = [z_vol.shape[3],z_vol.shape[4],z_vol.shape[2]]
    #注意下面这个window_size我也没办法一下子确定下来，主要是看原先的代码这么写，我们也就这么写
    transformer_block =  WindowEncoderSep(dim=z_vol_dim,input_resolution=input_resolution,num_heads=3,window_size=z_vol.shape[3]//2)
    if torch.cuda.is_available():
        transformer_block.cuda()
    transformed_z_vol = transformer_block(z_vol)

    #检查是否具有相同的shape
    assert transformed_z_vol.shape == z_vol.shape
    print("ji_transformed_z is")
    print(transformed_z_vol.shape)

    return transformed_z_vol


