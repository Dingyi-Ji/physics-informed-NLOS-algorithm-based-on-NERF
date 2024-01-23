# 将别的部分中的FK模块迁移到了这里

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ji_helper import definePsf, resamplingOperator, \
waveconvparam, waveconv

################################################################
class lct_fk(nn.Module):

    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, align_corners=False):
        super(lct_fk, self).__init__()

        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop

        self.bin_len = bin_len
        self.wall_size = wall_size

        self.align_corners = align_corners

        self.parpareparam()

    def change_bin_len(self, bin_len):
        print('change bin_len from %f to %f' % (self.bin_len, bin_len))

        self.bin_len = bin_len
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution

        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        M = temprol_grid
        N = sptial_grid

        fkrange = ((N * self.trange) / (M * self.width * 4)) ** 2
        gridznew = fkrange * self.gridxy_change + self.gridz_change
        gridznew = np.sqrt(gridznew)
        self.gridznew = torch.from_numpy(gridznew)

        newsame_1x2Mx2Nx2Nx1 = self.gridznew.unsqueeze(0).unsqueeze(4)
        newx = self.gridx_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        newy = self.gridy_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        self.newcoord_1x2Mx2Nx2Nx3 = torch.cat([newx, newy, newsame_1x2Mx2Nx2Nx1], dim=4)

        dnum = self.newcoord_dx2Mx2Nx2Nx3_todev.shape[0]
        dev = self.newcoord_dx2Mx2Nx2Nx3_todev.device
        self.newcoord_dx2Mx2Nx2Nx3_todev = self.newcoord_1x2Mx2Nx2Nx3.to(dev).repeat(dnum, 1, 1, 1, 1)
        self.gridznew_todev = self.gridznew.to(dev)

    #####################################################
    def parpareparam(self, ):

        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution

        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid

        ############################################################
        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))

        #######################################################
        zdim = np.arange(2 * temprol_grid, dtype=np.float32)
        xdim = np.arange(2 * sptial_grid, dtype=np.float32)

        zdim = (zdim - temprol_grid) / temprol_grid
        xdim = (xdim - sptial_grid) / sptial_grid
        ydim = xdim

        [gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M] = np.meshgrid(xdim, ydim, zdim)
        gridz_2Mx2Nx2N = np.transpose(gridz_2Nx2Nx2M, [2, 1, 0])
        gridy_2Mx2Nx2N = np.transpose(gridy_2Nx2Nx2M, [2, 1, 0])
        gridx_2Mx2Nx2N = np.transpose(gridx_2Nx2Nx2M, [2, 1, 0])

        '''
        print(gridz_2Mx2Nx2N[:, 0, 0])
        print(gridy_2Mx2Nx2N[0, :, 0])
        print(gridx_2Mx2Nx2N[0, 0, :])
        '''

        self.gridz_2Mx2Nx2N = torch.from_numpy(gridz_2Mx2Nx2N)
        self.gridy_2Mx2Nx2N = torch.from_numpy(gridy_2Mx2Nx2N)
        self.gridx_2Mx2Nx2N = torch.from_numpy(gridx_2Mx2Nx2N)

        self.gridxy_change = gridx_2Mx2Nx2N ** 2 + gridy_2Mx2Nx2N ** 2
        self.gridz_change = gridz_2Mx2Nx2N ** 2

        ###################################################
        M = temprol_grid
        N = sptial_grid

        fkrange = ((N * self.trange) / (M * self.width * 4)) ** 2
        gridznew = fkrange * self.gridxy_change + self.gridz_change
        gridznew = np.sqrt(gridznew)
        self.gridznew = torch.from_numpy(gridznew)

        newsame_1x2Mx2Nx2Nx1 = self.gridznew.unsqueeze(0).unsqueeze(4)
        newx = self.gridx_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        newy = self.gridy_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        self.newcoord_1x2Mx2Nx2Nx3 = torch.cat([newx, newy, newsame_1x2Mx2Nx2Nx1], dim=4)

        ####################################################
        self.xdim = xdim
        self.zdim = zdim
        self.z0pos = np.where(zdim > 0)[0][0]
        print('zzeropos %d' % self.z0pos)

    def todev(self, dev, dnum):
        self.gridz_1xMx1x1_todev = self.gridz_1xMx1x1.to(dev)
        self.gridz_square_1xMx1x1 = self.gridz_1xMx1x1_todev ** 2
        self.datapad_Dx2Tx2Hx2W = torch.zeros((dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid),
                                              dtype=torch.float32, device=dev)

        self.newcoord_dx2Mx2Nx2Nx3_todev = self.newcoord_1x2Mx2Nx2Nx3.to(dev).repeat(dnum, 1, 1, 1, 1)
        self.gridz_2Mx2Nx2N_todev = self.gridz_2Mx2Nx2N.to(dev)
        self.gridznew_todev = self.gridznew.to(dev)

    def roll_1(self, x_bxtxhxwx2, dim, n):
        if dim == 1:
            a = torch.cat((x_bxtxhxwx2[:, -n:], x_bxtxhxwx2[:, :-n]), dim=dim)
        if dim == 2:
            a = torch.cat((x_bxtxhxwx2[:, :, -n:], x_bxtxhxwx2[:, :, :-n]), dim=dim)
        if dim == 3:
            a = torch.cat((x_bxtxhxwx2[:, :, :, -n:], x_bxtxhxwx2[:, :, :, :-n]), dim=dim)
        return a

    def forward(self, feture_bxdxtxhxw, tbes, tens):
        print("success fk")
        #print("fk输入的维数{}".format(feture_bxdxtxhxw.shape))
        ###############################################
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        print(feture_bxdxtxhxw.shape)
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        dev = feture_bxdxtxhxw.device

        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32,
                                             device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)

        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop

        #################################################
        # step 0, pad data
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, self.crop, hnum, wnum)

        # c gridz_1xMx1x1 = self.gridz_1xMx1x1_todev
        # data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 2)
        gridz_square_1xMx1x1 = self.gridz_square_1xMx1x1
        data_BDxTxHxW = data_BDxTxHxW * gridz_square_1xMx1x1

        # numerical issue
        data_BDxTxHxW = F.relu(data_BDxTxHxW, inplace=False)
        data_BDxTxHxW = torch.sqrt(data_BDxTxHxW)

        # datapad_BDx2Tx2Hx2W = torch.zeros((bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_Dx2Tx2Hx2W = self.datapad_Dx2Tx2Hx2W
        # create new variable
        datapad_BDx2Tx2Hx2W = datapad_Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)

        datapad_BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = data_BDxTxHxW

        ###############################################
        # 1 fft
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        """这里进行一个修改，新版本中torch.rfft被torch.fft.rfftn所取代"""
        # datafre_BDX2Tx2Hx2Wx2 = torch.rfft(datapad_BDx2Tx2Hx2W, 3, onesided=False)
        datafre_temp = torch.fft.fftn(datapad_BDx2Tx2Hx2W, dim=(-3, -2, -1))
        datafre_BDX2Tx2Hx2Wx2 = torch.stack((datafre_temp.real, datafre_temp.imag), -1)

        # fftshift
        datafre_BDX2Tx2Hx2Wx2 = self.roll_1(datafre_BDX2Tx2Hx2Wx2, dim=1, n=temprol_grid)
        datafre_BDX2Tx2Hx2Wx2 = self.roll_1(datafre_BDX2Tx2Hx2Wx2, dim=2, n=sptial_grid)
        datafre_BDX2Tx2Hx2Wx2 = self.roll_1(datafre_BDX2Tx2Hx2Wx2, dim=3, n=sptial_grid)

        #########################################################
        # step2, ttrlt trick
        # simulate interpn
        # treat x and y as batch, sample z
        # shift

        if True:

            datafre_BDx2x2Hx2Wx2T = datafre_BDX2Tx2Hx2Wx2.permute(0, 4, 1, 2, 3)

            '''
            size = datafre_BDx2x2Hx2Wx2T.shape
            theta = torch.from_numpy(np.eye(3, 4, dtype=np.float32)).unsqueeze(0)
            gridstmp = F.affine_grid(theta, size, align_corners=self.align_corners)
            x = gridstmp[:, :, :, :, 0:1]
            y = gridstmp[:, :, :, :, 1:2]
            z = gridstmp[:, :, :, :, 2:3]
            '''

            newcoord_BDx2Mx2Nx2Nx3 = self.newcoord_dx2Mx2Nx2Nx3_todev.repeat(bnum, 1, 1, 1, 1)

            if True:
                datafrenew = F.grid_sample(datafre_BDx2x2Hx2Wx2T, newcoord_BDx2Mx2Nx2Nx3, \
                                           mode='bilinear', padding_mode='zeros', \
                                           align_corners=self.align_corners)
            else:
                datafrenew = F.grid_sample(datafre_BDx2x2Hx2Wx2T, newcoord_BDx2Mx2Nx2Nx3, \
                                           mode='bilinear', padding_mode='zeros')

            tdata_BDx2Tx2Hx2Wx2 = datafrenew.permute(0, 2, 3, 4, 1)
            tdata_BDx2Tx2Hx2Wx2 = tdata_BDx2Tx2Hx2Wx2.contiguous()

        ############################################################
        # actually, pytorch sampling will lead a little different
        else:
            import scipy.interpolate as si
            zdim = self.zdim
            xdim = self.xdim
            ydim = xdim

            gridznew = self.gridznew.numpy()
            gridy_2Mx2Nx2N = self.gridy_2Mx2Nx2N.numpy()
            gridx_2Mx2Nx2N = self.gridx_2Mx2Nx2N.numpy()

            datafre_bdxtxhxwx2 = datafre_BDX2Tx2Hx2Wx2.detach().cpu().numpy()
            datafre_bdxtxhxw = datafre_bdxtxhxwx2[:, :, :, :, 0] + 1j * datafre_bdxtxhxwx2[:, :, :, :, 1]

            re = []
            for datafre in datafre_bdxtxhxw:
                tvol = si.interpn(points=(zdim, ydim, xdim), values=datafre, \
                                  xi=np.stack([gridznew, gridy_2Mx2Nx2N, gridx_2Mx2Nx2N], axis=3), \
                                  method='linear', bounds_error=False, fill_value=0)
                re.append(tvol)

            re_bdxtxhxw = np.stack(re)
            re_real_bdxtxhxw = np.real(re_bdxtxhxw)
            re_imag_bdxtxhxw = np.imag(re_bdxtxhxw)

            re_real_bdxtxhxw = torch.from_numpy(re_real_bdxtxhxw).to(dev)
            re_imag_bdxtxhxw = torch.from_numpy(re_imag_bdxtxhxw).to(dev)
            tdata_BDx2Tx2Hx2Wx2 = torch.stack([re_real_bdxtxhxw, re_imag_bdxtxhxw], dim=4)

        #############################################################
        samplez_1xMxNxNx1 = self.gridz_2Mx2Nx2N_todev.unsqueeze(0).unsqueeze(4)
        sampleznew = self.gridznew_todev.unsqueeze(0).unsqueeze(4)

        tdata_BDx2Tx2Hx2Wx2[:, :self.z0pos, :, :, :] = 0
        tdata_BDx2Tx2Hx2Wx2 = tdata_BDx2Tx2Hx2Wx2 * samplez_1xMxNxNx1.abs()

        tdata_BDx2Tx2Hx2Wx2 = tdata_BDx2Tx2Hx2Wx2 / (sampleznew + 1e-8)

        ###########################################
        # ifft
        tdata_BDx2Tx2Hx2Wx2 = self.roll_1(tdata_BDx2Tx2Hx2Wx2, dim=1, n=temprol_grid)
        tdata_BDx2Tx2Hx2Wx2 = self.roll_1(tdata_BDx2Tx2Hx2Wx2, dim=2, n=sptial_grid)
        tdata_BDx2Tx2Hx2Wx2 = self.roll_1(tdata_BDx2Tx2Hx2Wx2, dim=3, n=sptial_grid)

        """这里也得修改"""
        ##data = torch.ifft(tdata_BDx2Tx2Hx2Wx2, 3)
        data_pre = torch.complex(tdata_BDx2Tx2Hx2Wx2[..., 0], tdata_BDx2Tx2Hx2Wx2[..., 1])
        data_temp = torch.fft.ifftn(data_pre, dim=(-3, -2, -1))
        data = torch.stack((data_temp.real, data_temp.imag), -1)

        data = data[:, :temprol_grid, :sptial_grid, :sptial_grid]
        data = data[:, :, :, :, 0] ** 2 + data[:, :, :, :, 1] ** 2;

        ##########################################################################3
        volumn_BDxTxHxW = data.view(bnum * dnum, self.crop, hnum, wnum)

        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)

        #print("fk输出的维数{}".format(volumn_BxDxTxHxW.shape))

        return volumn_BxDxTxHxW


class phasor(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5):
        super(phasor, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
        
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        
        virtual_cos_wave_k, virtual_sin_wave_k = \
        waveconvparam(bin_resolution, virtual_wavelength, cycles)
        
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0)
        
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        # lct
        # invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        # bp
        invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
    def todev(self, dev, dnum):
        
        self.virtual_cos_sin_wave_inv_2x1xk_todev = self.virtual_cos_sin_wave_inv_2x1xk.to(dev)
        self.datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)
        
    def forward(self, feture_bxdxtxhxw, tbes, tens):
        
        print("采用自己的phasor")
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        dev = feture_bxdxtxhxw.device
        
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum)
        
        ############################################################
        # Step 1: convolve measurement volume with virtual wave
        
        data_BDxHxWxT = data_BDxTxHxW.permute(0, 2, 3, 1)
        data_BDHWx1xT = data_BDxHxWxT.reshape(-1, 1, tnum)
        knum = self.virtual_cos_sin_wave_inv_2x1xk.shape[2]
        phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev, padding=knum // 2)
        if knum % 2 == 0:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T[:, :, 1:]
        else:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T
        
        data_BDxHxWx2xT = data_BDHWx2xT.reshape(bnum * dnum, hnum, wnum, 2, tnum)
        data_2xBDxTxHxW = data_BDxHxWx2xT.permute(3, 0, 4, 1, 2)
        data_2BDxTxHxW = data_2xBDxTxHxW.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        
        # datapad_2BDx2Tx2Hx2W = torch.zeros((2 * bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        # create new variable
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        # actually, because it is all zero so it is ok
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W
        
        left = self.mtx_MxM_todev
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        
        ###########################################################3
        # Step 3: convolve with backprojection kernel
        
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        # datafre = torch.rfft(datapad_2BDx2Tx2Hx2W, 3, onesided=False)   # pytorch 1.6
        datafre_temp = torch.fft.fftn(datapad_2BDx2Tx2Hx2W, dim=(-3, -2, -1))
        # datafre = torch.stack((datafre_temp.real, datafre_temp.imag), -1)

        datafre_real = datafre_temp.real  # datafre[:, :, :, :, 0]
        datafre_imag = datafre_temp.imag  # datafre[:, :, :, :, 1]

        re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
        re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
        #refre = torch.stack([re_real, re_imag], dim=4)
        # re = torch.ifft(refre, 3)   # pytorch 1.6
        refre = torch.complex(re_real, re_imag)
        re_temp = torch.fft.ifftn(refre, dim=(-3, -2, -1))
        re = torch.stack((re_temp.real, re_temp.imag), -1)

        volumn_2BDxTxHxWx2 = re[:, :temprol_grid, :sptial_grid, :sptial_grid, :]
        
        ########################################################################
        # Step 4: compute phasor field magnitude and inverse LCT
        
        cos_real = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 0]
        cos_imag = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 1]
        
        sin_real = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 0]
        sin_imag = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 1]
        
        sum_real = cos_real ** 2 - cos_imag ** 2 + sin_real ** 2 - sin_imag ** 2
        sum_image = 2 * cos_real * cos_imag + 2 * sin_real * sin_imag
        
        tmp = (torch.sqrt(sum_real ** 2 + sum_image ** 2) + sum_real) / 2
        # numerical issue
        tmp = F.relu(tmp, inplace=False)
        sqrt_sum_real = torch.sqrt(tmp)
        
        #####################################################################
        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        ########################################################################
        # do we force to be > 0?
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        
        return volumn_BxDxTxHxW




class FK(phasor): # 继承自RSDBase父类
    """
    Args:
        x (float tensor, (bs, c, t, h, w)): input time-domain features.
        sqrt (bool): if True, take the square root before normalization.

    Returns:
        x (float tensor, (bs, c, d, h, w)): output space-domain features.
    """
    def __init__(self, **kwargs):
        super(FK, self).__init__(**kwargs) # 从父类继承初始化

class Phasor(phasor): # 继承自RSDBase父类
    """
    Args:
        x (float tensor, (bs, c, t, h, w)): input time-domain features.
        sqrt (bool): if True, take the square root before normalization.

    Returns:
        x (float tensor, (bs, c, d, h, w)): output space-domain features.
    """
    def __init__(self, **kwargs):
        super(Phasor, self).__init__(**kwargs) # 从父类继承初始化

class FKEfficient(phasor):
    def __init__(self, **kwargs):
        # 注意，这里目前还处于没修改的状态，就让FK和FKEfficient共用一个类，不做其他修改
        super(FKEfficient, self).__init__(**kwargs)

