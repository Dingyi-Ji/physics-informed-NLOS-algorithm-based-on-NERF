import json
import numpy as np
# import pdb
import torch

#from ray_utils import get_rays, get_ray_directions, get_ndc_rays   # 暂时不用那几个函数，这里就先注释掉了


BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')  # 维度(1, 8, 3)


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2  表示以2为底的对数，log2_hashmap_size表示哈希表大小的对数，以2为底进行计算
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    """哈希函数中，使用素数作为乘法因子可以帮助增加哈希函数的随机性和分散性，从而减少哈希冲突的可能性。
    在给定的代码中，primes 列表中的数字是经过精心选择的素数，用于与坐标的对应维度进行乘法运算。
    通过使用不同的素数乘法因子，可以在哈希计算中引入更多的随机性，从而增加哈希值的独特性和分布性。
    因此，使用素数作为乘法因子是一种常见的技巧，用于构造高质量的哈希函数，以减少哈希冲突并提高哈希表的性能。
    因为他最多支持7个维度，所以这里写的也是七个素数"""

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]
    """torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result 执行以下操作：

    (1<<log2_hashmap_size) 表示将1左移 log2_hashmap_size 位，相当于计算 2 的 log2_hashmap_size 次方。
    -1 表示对上一步结果减去 1，得到 2^log2_hashmap_size - 1。
    torch.tensor(...) 将上一步的结果转换为张量。
    .to(xor_result.device) 将上一步的张量移到与 xor_result 张量相同的设备上，以保持设备一致性。
    & xor_result 表示对上一步的张量和 xor_result 进行按位与操作。
    整体来说，上述表达式的目的是将 xor_result 与哈希表大小进行取模，得到一个在哈希表索引范围内的哈希值。
    这样做可以确保哈希值适应哈希表的大小，并产生一个有效的索引值"""

    """
    纪注：当除数是2^n时候（假如被除数是s，除数是m），则可以按照s&(m-1)实现取模运算，哈希表的大小一定是2的幂次方大小
    返回的是 坐标映射到哈希表中的索引
    """
    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

#
# def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
#     """
#     用于获取在3D空间中包围Blender对象的边界框
#     arg:
#         camera_transforms：包含相机变换信息的字典。
#         H：图像的高度。
#         W：图像的宽度。
#         near：近裁剪面的距离，默认为2.0。
#         far：远裁剪面的距离，默认为6.0。
#
#     """
#     """
#     计算相机的焦距 (focal)。通过使用相机水平视角 (camera_angle_x) 的一半，结合图像宽度和焦距的关系，计算焦距的值。
#     获取相机坐标系下的射线方向 (directions)，通过调用 get_ray_directions 函数。
#     初始化最小边界 (min_bound) 和最大边界 (max_bound)，分别设置为一个较大的初始值和一个较小的初始值。
#     创建一个空列表 points，用于存储边界框的顶点。
#     遍历相机变换中的每一帧：
#     获取当前帧的相机到世界坐标系的变换矩阵 (c2w)。
#     使用射线方向和相机到世界变换矩阵，计算光线的原点 (rays_o) 和方向 (rays_d)。
#     定义一个内部函数 find_min_max，用于更新最小边界和最大边界的值。
#     对于边界框的四个角点，计算最小点 (min_point) 和最大点 (max_point)，并将其添加到 points 列表中。
#     调用 find_min_max 函数，更新最小边界和最大边界的值。
#     计算最终的边界框范围，通过从最小边界中减去 [1.0, 1.0, 1.0]，以及从最大边界中加上 [1.0, 1.0, 1.0]。
#     返回计算得到的边界框范围。"""
#
#     camera_angle_x = float(camera_transforms['camera_angle_x'])
#     focal = 0.5*W/np.tan(0.5 * camera_angle_x)
#
#     # ray directions in camera coordinates
#     directions = get_ray_directions(H, W, focal)
#
#     min_bound = [100, 100, 100]
#     max_bound = [-100, -100, -100]
#
#     points = []
#
#     for frame in camera_transforms["frames"]:
#         c2w = torch.FloatTensor(frame["transform_matrix"])
#         rays_o, rays_d = get_rays(directions, c2w)
#
#         def find_min_max(pt):
#             # 这里实际上就是在确定完究竟有哪些点以后（根据射线和近远面的距离），去判断每个维度上的最远/近点应该在哪里
#             for i in range(3):
#                 if(min_bound[i] > pt[i]):
#                     min_bound[i] = pt[i]
#                 if(max_bound[i] < pt[i]):
#                     max_bound[i] = pt[i]
#             return
#
#         for i in [0, W-1, H*W-W, H*W-1]:
#             min_point = rays_o[i] + near*rays_d[i]
#             max_point = rays_o[i] + far*rays_d[i]
#             points += [min_point, max_point]
#             find_min_max(min_point)
#             find_min_max(max_point)
#
#     return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))
#
#
# def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0):
#     H, W, focal = hwf
#     H, W = int(H), int(W)
#
#     # ray directions in camera coordinates
#     directions = get_ray_directions(H, W, focal)
#
#     min_bound = [100, 100, 100]
#     max_bound = [-100, -100, -100]
#
#     points = []
#     poses = torch.FloatTensor(poses)
#     for pose in poses:
#         rays_o, rays_d = get_rays(directions, pose)
#         rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)
#
#         def find_min_max(pt):
#             for i in range(3):
#                 if(min_bound[i] > pt[i]):
#                     min_bound[i] = pt[i]
#                 if(max_bound[i] < pt[i]):
#                     max_bound[i] = pt[i]
#             return
#
#         for i in [0, W-1, H*W-W, H*W-1]:
#             min_point = rays_o[i] + near*rays_d[i]
#             max_point = rays_o[i] + far*rays_d[i]
#             points += [min_point, max_point]
#             find_min_max(min_point)
#             find_min_max(max_point)
#
#     return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    用于获取体素的顶点
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    log2_hashmap_size：log2_hashmap_size表示哈希表大小的对数，以2为底进行计算
    '''
    box_min, box_max = bounding_box # 先获取边界

    #print("在哪个设备上")
    #print(xyz.device)
    #print(box_max.device)
    #tmp = torch.min(xyz, box_max)

    keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min) # 这里面的minmax就是为了将坐标约束到min到max中间
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        # torch.clamp用于对张量进行逐元素的截断操作
        #print(box_min)

        # 这里不能直接将box_min加进来，因为clamp(): argument 'min' must be Number, not Tensor
        box_min = box_min
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

        

    grid_size = (box_max-box_min)/resolution    # 在min到max之间有多少个点，决定了每两个点之间的距离（resolution）
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()    # 获取的是某点相对于边界最小点的索引
    voxel_min_vertex = bottom_left_idx*grid_size + box_min  # 这里的一通操作实际上是为了取整，也就是为了找离他最近的左下角顶点
    #print("grid size device {}".format(grid_size.device))
    #print("voxel_max_vertex device {}".format(voxel_min_vertex.device))
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0]).cuda()*grid_size # 那不用说，在这个格子中，离他最远的顶点就加1
    # （或者叫临近的那个顶点，实际上也就是为了插值来使用的）


    ###### ？？？？？？？？？？？ 这里我不知道他为啥加了个偏置进来
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask



if __name__=="__main__":
    with open("data/nerf_synthetic/chair/transforms_train.json", "r") as f:
        camera_transforms = json.load(f)
    
    # bounding_box = get_bbox3d_for_blenderobj(camera_transforms, 800, 800)  # 用不到的都先被我注释掉了
