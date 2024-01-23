基于物理的非视域成像（NLOS）算法，基于NERF

光路原理图与算法示意图
![method](https://github.com/Dingyi-Ji/physics-informed-NLOS-algorithm-based-on-NERF/assets/59365251/3e38db56-6896-4bbd-91fa-43c9cb95993a)

Nerf引入解决NLOS问题的思想最早来自下面的文章,NeRF 假设体渲染模型，并着手使用多层感知 (MLP) 来恢复每体素场景密度和每方向颜色,相比于其他深度学习方法，对于解决NLOS问题有着优势
Non-line-of-sight Imaging via Neural Transient Fields
注：NeRF实际上就是用了隐式神经表示的方法，将三维结构数据隐含在全连接神经网络MLP的参数当中，输入5D数据输出该(x,y,z)点的颜色和体密度数据。利用可微渲染器（也就是一个可微的前向模型）计算所获得3D数据的在对应视角下的2D投影，与对应的2D图像计算损失值，实现对于网络参数的训练
![image](https://github.com/Dingyi-Ji/physics-informed-NLOS-algorithm-based-on-NERF/assets/59365251/dd0fd3f5-62fa-44dc-b016-dad14268f1b6)


融合物理算法，将原始数据从xyt维度转换到xyt维度，提高算法表现能力和物理可解释性
该思想在非视域成像领域的原创来自于下文
W. Chen, F. Wei, K. N. Kutulakos, S. Rusinkiewicz, and F. Heide, “Learned feature embeddings for non-line-of-sight imaging and recognition,” ACM Transactions on Graphics (ToG), 2020.
将该思想与Nerf相融合的工作来自于下文，我们即在该工作的框架上进行深入
Mu F, Mo S, Peng J, et al. Physics to the rescue: Deep non-line-of-sight reconstruction for high-speed imaging[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.


引入条件nerf，并改用instant-ngp类似的hash编码方式，加快训练和推理速度，并在网络中保存了相关训练参数，条件nerf的构建参考pixelNERF,IBRnet以及该类其他算法，主要是为了学习场景先验
采用instant-ngp是为了利用多分辨哈希编码，减小网络规模，减少哈希碰撞；同时多分辨哈希编码使得坐标信息映射到可学习的特征向量，加快训练速度
之所以能加快训练速度，原因在于MLP每次需要更新整个网络，但特征向量只会有很少一部分受到影响，因为每个特征向量都是由相近的八个点插值得到的，因此只跟这8个点有关，所以更新更加迅速，属于是内存换时间
NGP选取了参数化的voxel grid作为场景表达。通过学习，让voxel中保存的参数成为场景密度的形状
Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelNeRF: Neural radiance fields from one or few images, 2020
Q. Wang et al., "IBRNet: Learning Multi-View Image-Based Rendering," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp. 4688-4697, doi: 10.1109/CVPR46437.2021.00466
![image](https://github.com/Dingyi-Ji/physics-informed-NLOS-algorithm-based-on-NERF/assets/59365251/ff357ee5-67d5-4a9b-ace3-dd7fc765f198)
![image](https://github.com/Dingyi-Ji/physics-informed-NLOS-algorithm-based-on-NERF/assets/59365251/78a327e4-7fa9-4cd2-8f9c-ece311c08c10)


采用CBAM实现条件nerf的条件输出，同时对于时间（t维度，在经过变换后为z维度）和空间维度（xy维度）信息的全局与局部感知
Woo S, Park J, Lee J Y, et al. Cbam: Convolutional block attention module[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 3-19.
![image](https://github.com/Dingyi-Ji/physics-informed-NLOS-algorithm-based-on-NERF/assets/59365251/666e9e01-7ade-4887-8b84-e135af249bf9)
![image](https://github.com/Dingyi-Ji/physics-informed-NLOS-algorithm-based-on-NERF/assets/59365251/8d150f6b-4d21-45de-bc59-5141c0c200ce)


