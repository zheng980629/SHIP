import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from thop import profile
import time

@MODEL_REGISTRY.register()
class FusionMaskModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(FusionMaskModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # from thop import profile
        # from thop import clever_format
        # dummy_inf = torch.rand(1, 3, 512, 512).cuda()
        # dummy_vis = torch.rand(1, 3, 512, 512).cuda()
        # flops, params = profile(self.net_g, inputs=(dummy_vis, dummy_inf))
        # flops, params = clever_format([flops, params], "%.3f")
        # print((f'The FLOPs of the model is: {flops}'))
        # exit(0)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('max_opt'):
            self.cri_max = build_loss(train_opt['max_opt']).to(self.device)
        else:
            self.cri_max = None

        if train_opt.get('gradient_opt'):
            self.cri_gradient = build_loss(train_opt['gradient_opt']).to(self.device)
        else:
            self.cri_gradient = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('mutual_opt'):
            self.cri_mutual = build_loss(train_opt['mutual_opt']).to(self.device)
        else:
            self.cri_mutual = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def RGB2YCrCb(self, input_im):
        im_flat = input_im.transpose(1, 3).transpose(
            1, 2).reshape(-1, 3)  # (nhw,c)
        R = im_flat[:, 0]
        G = im_flat[:, 1]
        B = im_flat[:, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 0.5
        Cb = (B - Y) * 0.564 + 0.5
        Y = torch.unsqueeze(Y, 1)
        Cr = torch.unsqueeze(Cr, 1)
        Cb = torch.unsqueeze(Cb, 1)
        temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
        out = (
            temp.reshape(
                list(input_im.size())[0],
                list(input_im.size())[2],
                list(input_im.size())[3],
                3,
            )
            .transpose(1, 3)
            .transpose(2, 3)
        )
        return out

    def YCrCb2RGB(self, input_im):
        im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
        mat = torch.tensor(
            [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
        ).cuda()
        bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
        temp = (im_flat + bias).mm(mat).cuda()
        out = (
            temp.reshape(
                list(input_im.size())[0],
                list(input_im.size())[2],
                list(input_im.size())[3],
                3,
            )
            .transpose(1, 3)
            .transpose(2, 3)
        )
        return out

    def feed_data(self, data):
        self.vi = data['lq'].to(self.device)
        self.ir = data['gt'].to(self.device)[:, :1]
        self.mask = data['mask'].to(self.device)[:, :1]

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.vi_ycrcb = self.RGB2YCrCb(self.vi)

        # self.vi_ycrcb, self.ir = torch.randn(1 ,3, 448, 620).cuda(), torch.randn(1 ,1, 448, 620).cuda()
        # start = time.time()
        self.output_y = self.net_g(self.vi_ycrcb, self.ir)
        
        # flops, params = profile(self.net_g, inputs=(self.vi_ycrcb, self.ir))
        # print("flops",flops)
        # print("params",params)
        # end = time.time()
        # print("time: ", end-start)

        self.output_y = torch.clamp(self.output_y, 0, 1)

        self.output_ycrcb = torch.cat(
                (self.output_y, self.vi_ycrcb[:, 1:2, :, :],
                 self.vi_ycrcb[:, 2:, :, :]),
                dim=1,
            )
        self.output = self.YCrCb2RGB(self.output_ycrcb)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.vi_ycrcb, self.ir, self.output_ycrcb, self.mask)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_max:
            l_max = self.cri_max(self.vi_ycrcb, self.ir, self.output_ycrcb)
            l_total += l_max
            loss_dict['l_max'] = l_max

        if self.cri_gradient:
            l_gradient = self.cri_gradient(self.vi_ycrcb, self.ir, self.output_ycrcb)
            l_total += l_gradient
            loss_dict['l_gradient'] = l_gradient

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.vi_ycrcb, self.ir, self.output_ycrcb)
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim

        # if self.cri_mutual:
        #     l_mtual = self.cri_mutual(self.panf, self.mHRf)
        #     l_total += l_mtual
        #     loss_dict['l_mtual'] = l_mtual

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.vi_ycrcb = self.RGB2YCrCb(self.vi)
                self.output_y  = self.net_g_ema(self.vi_ycrcb, self.ir)
                self.output_y = torch.clamp(self.output_y, 0, 1)

                self.output_ycrcb = torch.cat(
                        (self.output_y, self.vi_ycrcb[:, 1:2, :, :],
                        self.vi_ycrcb[:, 2:, :, :]),
                        dim=1,
                    )
                self.output_test = self.YCrCb2RGB(self.output_ycrcb)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.vi_ycrcb = self.RGB2YCrCb(self.vi)
                self.output_y = self.net_g(self.vi_ycrcb, self.ir)
                self.output_y = torch.clamp(self.output_y, 0, 1)

                self.output_ycrcb = torch.cat(
                        (self.output_y, self.vi_ycrcb[:, 1:2, :, :],
                        self.vi_ycrcb[:, 2:, :, :]),
                        dim=1,
                    )
                self.output_test = self.YCrCb2RGB(self.output_ycrcb)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # [0-255]
            vis_img = tensor2img([visuals['vi']])
            ir_img = tensor2img([visuals['ir']])
            mask_img = tensor2img([visuals['mask']])
            fused_img = tensor2img([visuals['fused']])

            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path_vi = osp.join(self.opt['path']['visualization'],
                                             f'{img_name}_{current_iter}_vi.png')
                    save_img_path_ir = osp.join(self.opt['path']['visualization'],
                                             f'{img_name}_{current_iter}_ir.png')
                    save_img_path_mask = osp.join(self.opt['path']['visualization'],
                                             f'{img_name}_{current_iter}_mask.png')
                    save_img_path_fused = osp.join(self.opt['path']['visualization'],
                                                 f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path_vi = osp.join(self.opt['path']['visualization'],
                                                f'{img_name}_{current_iter}_vi.png')
                        save_img_path_ir = osp.join(self.opt['path']['visualization'],
                                                f'{img_name}_{current_iter}_ir.png')
                        save_img_path_mask = osp.join(self.opt['path']['visualization'],
                                                f'{img_name}_{current_iter}_mask.png')
                        save_img_path_fused = osp.join(self.opt['path']['visualization'],
                                                 f'{img_name}.png')
                    else:
                        save_img_path_fused = osp.join(self.opt['path']['visualization'],
                                                 f'{img_name}.png')
                        
                # imwrite(vis_img, save_img_path_vi)
                # imwrite(ir_img, save_img_path_ir)
                # imwrite(mask_img, save_img_path_mask)
                imwrite(fused_img, save_img_path_fused)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['vi'] = self.vi.detach().cpu()
        out_dict['ir'] = self.ir.detach().cpu()
        out_dict['mask'] = self.mask.detach().cpu()
        out_dict['fused'] = self.output_test.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
