print(">>>> LOADED RGRG:", __file__)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

import lpips
import torchvision
import time


class RGRG():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 2:
            model_name = 'inpaint'
        else:
            model_name = 'unknown'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.transf = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        # wandb 开关（由 config.USE_WANDB 控制 + 实际可用性）
        self.use_wandb = bool(getattr(self.config, "USE_WANDB", False) and _WANDB_AVAILABLE)

        # train dataset
        if self.config.MODE == 1 and self.config.MODEL == 2:
            self.train_dataset = Dataset(
                config,
                config.TRAIN_INPAINT_IMAGE_FLIST,
                config.TRAIN_MASK_FLIST,
                augment=True,
                training=True,
                ref_flist=getattr(self.config, "STYLE_REF_FLIST", None),  # ✨ 新增
            )

        # test dataset
        if self.config.MODE == 2 and self.config.MODEL == 2:
            print('model == 2')
            self.test_dataset = Dataset(
                config,
                config.TEST_INPAINT_IMAGE_FLIST,
                config.TEST_MASK_FLIST,
                augment=False,
                training=False,
                ref_flist=getattr(self.config, "STYLE_REF_FLIST", None),  # ✅ 加这一行
            )

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if getattr(config, "DEBUG", 0) != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

        self.vis_idx = 0
    def load(self):
        if self.config.MODEL == 2:
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 2:
            self.inpaint_model.save()


    def train(self):

        if self.use_wandb:
            try:
                has_run = hasattr(wandb, "run") and (wandb.run is not None)
            except Exception:
                has_run = False
            if has_run:
                wandb.watch(self.inpaint_model, log='all', log_freq=10)


        print(">>> len(train_dataset):", len(self.train_dataset))
        bs = min(self.config.BATCH_SIZE, max(1, len(self.train_dataset)))
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=bs,
            num_workers=0,
            drop_last=False,
            shuffle=True
        )
        print(">>> effective batch_size:", bs)
        print(">>> len(train_loader):", len(train_loader))
        if len(train_loader) == 0:
            raise RuntimeError("Empty train_loader.")

        # debug 目录
        create_dir(self.results_path)
        path_masked_dbg = os.path.join(self.results_path, self.model_name, 'masked_debug')
        path_result_dbg = os.path.join(self.results_path, self.model_name, 'result_debug')
        path_joint_dbg = os.path.join(self.results_path, self.model_name, 'joint_debug')
        create_dir(path_masked_dbg);
        create_dir(path_result_dbg);
        create_dir(path_joint_dbg)

        epoch = 0
        keep_training = True
        model_flag = self.config.MODEL
        max_iteration = int(float(self.config.MAX_ITERS))
        total = len(self.train_dataset)


        while keep_training:
            epoch += 1
            print("\n\nTraining epoch:", epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            # 保存用的 batch 样本（最后一个 batch）
            last_images = None
            last_outputs_img = None
            last_outputs_merged = None
            last_masks = None

            for bidx, items in enumerate(train_loader):
                self.inpaint_model.train()

    
                if model_flag == 2:
                    if len(items) == 3:
                        images, masks, image_ref = self.cuda(*items)
                    else:
                        images, masks = self.cuda(*items)
                        image_ref = None

                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, \
                        gen_content_loss, gen_style_loss = \
                        self.inpaint_model.process(images, masks, image_ref=image_ref)

                    outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                    # PSNR + MAE
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = torch.mean(torch.abs(images - outputs_merged)).float()

                    psnr_value = float(psnr) if not isinstance(psnr, torch.Tensor) else psnr.item()
                    mae_value = float(mae) if not isinstance(mae, torch.Tensor) else mae.item()

                    logs.append(("psnr", psnr_value))
                    logs.append(("mae", mae_value))

       
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                    if iteration % self.config.SAVE_INTERVAL == 0:
                        print(f"[Checkpoint] saving at iter {iteration} ...")
                        self.save()
         
                    if bidx == 0 or iteration % 50 == 0:
                        print(f"  iter={iteration} | gLoss={float(gen_loss.detach().cpu()):.4f} "
                              f"dLoss={float(dis_loss.detach().cpu()):.4f}")

     
                    if epoch == 1 and bidx == 0:
                        inputs = (images * (1 - masks))
                        images_joint = stitch_images(
                            self.postprocess(images),
                            self.postprocess(inputs),
                            self.postprocess(outputs_img),
                            self.postprocess(outputs_merged),
                            img_per_row=1
                        )

                        self.vis_idx += 1
                        dbg_name = f"{self.vis_idx:06d}.png"
                        masked_images = self.postprocess(images * (1 - masks) + masks)[0]
                        images_result = self.postprocess(outputs_merged)[0]

                        images_joint.save(os.path.join(path_joint_dbg, dbg_name))
                        imsave(masked_images, os.path.join(path_masked_dbg, dbg_name))
                        imsave(images_result, os.path.join(path_result_dbg, dbg_name))

                        print("  [DEBUG SAVE OK]:", os.path.join(path_joint_dbg, dbg_name))

                    if iteration >= max_iteration:
                        keep_training = False
                        break

              
                    logs = [("epoch", epoch), ("iter", iteration)] + logs
                    progbar.add(len(images),
                                values=logs if self.config.VERBOSE else
                                [x for x in logs if not x[0].startswith('l_')])

            
                    if iteration % 10 == 0 and self.use_wandb:
                        try:
                            wandb.log({
                                'gen_loss': float(gen_loss.detach().cpu()),
                                'l1_loss': float(gen_l1_loss.detach().cpu()),
                                'style_loss': float(gen_style_loss.detach().cpu()),
                                'perceptual_loss': float(gen_content_loss.detach().cpu()),
                                'gen_gan_loss': float(gen_gan_loss.detach().cpu()),
                                'dis_loss': float(dis_loss.detach().cpu())
                            }, step=iteration)
                        except:
                            pass

         
                    last_images = images
                    last_outputs_img = outputs_img
                    last_outputs_merged = outputs_merged
                    last_masks = masks



            print(f"\n[Epoch {epoch}] Saving sample preview...")
            self.save()  
            path_masked = os.path.join(self.results_path, self.model_name, "masked_epoch")
            path_result = os.path.join(self.results_path, self.model_name, "result_epoch")
            path_joint = os.path.join(self.results_path, self.model_name, "joint_epoch")
            create_dir(path_masked);
            create_dir(path_result);
            create_dir(path_joint)

            name = f"epoch_{epoch:03d}.png"

            inputs = (last_images * (1 - last_masks))
            images_joint = stitch_images(
                self.postprocess(last_images),
                self.postprocess(inputs),
                self.postprocess(last_outputs_img),
                self.postprocess(last_outputs_merged),
                img_per_row=1
            )

            masked_images = self.postprocess(last_images * (1 - last_masks) + last_masks)[0]
            images_result = self.postprocess(last_outputs_merged)[0]

            images_joint.save(os.path.join(path_joint, name))
            imsave(masked_images, os.path.join(path_masked, name))
            imsave(images_result, os.path.join(path_result, name))

            print("  [EPOCH SAVE OK]:", os.path.join(path_joint, name))


        print("\nEnd training....")

    def test(self):
        import csv
        import time
        import os
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        self.inpaint_model.eval()
        model_flag = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        psnr_list, ssim_list, l1_list, lpips_list = [], [], [], []

        psnr_mis_list, ssim_mis_list, l1_mis_list, lpips_mis_list = [], [], [], []


        stats_rows = []
        sim_vals, rel_vals, gate_vals = [], [], []
        sim_mis_vals, rel_mis_vals, gate_mis_vals = [], [], []

        print('here')
        index = 0

        def _mean_or_none(t):
            if t is None:
                return None
            try:
                return float(t.mean().item())
            except Exception:
                return None

        def _sample_mismatch_ref(cur_index: int):
            if len(self.test_dataset) <= 1:
                return None
            for _ in range(20):
                ridx = int(np.random.randint(0, len(self.test_dataset)))
                if ridx == cur_index:
                    continue
                it2 = self.test_dataset[ridx]  # (img, mask, ref)
                if isinstance(it2, (list, tuple)) and len(it2) == 3:
                    _, _, ref = it2
                    return ref
            return None


        def _strong_mismatch_transform(ref_bchw):
            """
            ref_bchw: (B,3,H,W), expected in [0,1]
            """
            x = ref_bchw.clone()

            x = 1.0 - x


            x = x[:, [2, 0, 1], :, :]

            x = x + 0.15 * torch.randn_like(x)

            shift = (torch.rand(x.size(0), 3, 1, 1, device=x.device, dtype=x.dtype) - 0.5) * 0.6
            x = x + shift

            return torch.clamp(x, 0.0, 1.0)


        psnr_drop_list, ssim_drop_list, l1_drop_list, lpips_drop_list = [], [], [], []
        gate_for_drop = []  
        for cur_idx, items in enumerate(test_loader):

            if isinstance(items, (list, tuple)) and len(items) == 3:
                images, masks, image_ref = self.cuda(*items)
            else:
                images, masks = self.cuda(*items)
                image_ref = None

            index += 1

            if torch.isnan(images).any() or torch.isnan(masks).any():
                print("[WARN] NaN detected in inputs, skipping batch")
                continue

            if model_flag != 2:
                continue

            inputs = (images * (1 - masks))

            # =============== forward ===============
            with torch.no_grad():
                tsince = int(round(time.time() * 1000))

                # ---------- matched ----------
                outputs_img = self.inpaint_model(images, masks, image_ref=image_ref)
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))


                g = self.inpaint_model.generator
                sim_mean = _mean_or_none(getattr(g, "last_sim", None))
                rel_mean = _mean_or_none(getattr(g, "last_rel", None))
                gate_mean = _mean_or_none(getattr(g, "last_gate", None))

                # ---------- mismatched ----------
                outputs_img_mis = None
                outputs_merged_mis = None
                sim_mis_mean = rel_mis_mean = gate_mis_mean = None

                if image_ref is not None:
                    ref_mis = _sample_mismatch_ref(cur_idx)
                    if ref_mis is not None and torch.is_tensor(ref_mis):
                        if ref_mis.dim() == 3:
                            ref_mis = ref_mis.unsqueeze(0)  # (1,3,H,W)

                        ref_mis = ref_mis.to(images.device)

                        
                        ref_mis = torch.clamp(ref_mis, 0.0, 1.0)

                        ref_mis = _strong_mismatch_transform(ref_mis)

                        outputs_img_mis = self.inpaint_model(images, masks, image_ref=ref_mis)
                        outputs_merged_mis = (outputs_img_mis * masks) + (images * (1 - masks))

                        g2 = self.inpaint_model.generator
                        sim_mis_mean = _mean_or_none(getattr(g2, "last_sim", None))
                        rel_mis_mean = _mean_or_none(getattr(g2, "last_rel", None))
                        gate_mis_mean = _mean_or_none(getattr(g2, "last_gate", None))

                ttime_elapsed = int(round(time.time() * 1000)) - tsince
                print('test time elaspsed {}ms'.format(ttime_elapsed))

            # =============== metrics: matched ===============
            psnr, ssim = self.metric(images, outputs_merged)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if torch.cuda.is_available():
                pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(),
                                      self.transf(images[0].cpu()).cuda()).item()
            else:
                pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()),
                                      self.transf(images[0].cpu())).item()
            lpips_list.append(pl)

            l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
            l1_list.append(l1_loss)


            psnr_mis = ssim_mis = pl_mis = l1_mis = None
            if outputs_merged_mis is not None:
                psnr_mis, ssim_mis = self.metric(images, outputs_merged_mis)
                psnr_mis_list.append(psnr_mis)
                ssim_mis_list.append(ssim_mis)

                if torch.cuda.is_available():
                    pl_mis = self.loss_fn_vgg(self.transf(outputs_merged_mis[0].cpu()).cuda(),
                                              self.transf(images[0].cpu()).cuda()).item()
                else:
                    pl_mis = self.loss_fn_vgg(self.transf(outputs_merged_mis[0].cpu()),
                                              self.transf(images[0].cpu())).item()
                lpips_mis_list.append(pl_mis)

                l1_mis = torch.nn.functional.l1_loss(outputs_merged_mis, images, reduction='mean').item()
                l1_mis_list.append(l1_mis)

                # drop（mismatch - matched）
                psnr_drop_list.append(float(psnr_mis - psnr))
                ssim_drop_list.append(float(ssim_mis - ssim))
                l1_drop_list.append(float(l1_mis - l1_loss))
                lpips_drop_list.append(float(pl_mis - pl))

                if gate_mis_mean is not None:
                    gate_for_drop.append(float(gate_mis_mean))

            msg = (
                f"[Matched] psnr:{psnr:.3f}/{np.average(psnr_list):.3f}  "
                f"ssim:{ssim:.4f}/{np.average(ssim_list):.4f}  "
                f"l1:{l1_loss:.5f}/{np.average(l1_list):.5f}  "
                f"lpips:{pl:.5f}/{np.average(lpips_list):.5f}  "
                f"| sim:{sim_mean} rel:{rel_mean} gate:{gate_mean}"
            )
            if psnr_mis is not None:
                msg += (
                    f" || [Mismatch*] psnr:{psnr_mis:.3f}/{np.average(psnr_mis_list):.3f}  "
                    f"ssim:{ssim_mis:.4f}/{np.average(ssim_mis_list):.4f}  "
                    f"l1:{l1_mis:.5f}/{np.average(l1_mis_list):.5f}  "
                    f"lpips:{pl_mis:.5f}/{np.average(lpips_mis_list):.5f}  "
                    f"| sim:{sim_mis_mean} rel:{rel_mis_mean} gate:{gate_mis_mean}"
                )
            print(msg)

            # =============== record stats ===============
            if sim_mean is not None: sim_vals.append(sim_mean)
            if rel_mean is not None: rel_vals.append(rel_mean)
            if gate_mean is not None: gate_vals.append(gate_mean)

            if sim_mis_mean is not None: sim_mis_vals.append(sim_mis_mean)
            if rel_mis_mean is not None: rel_mis_vals.append(rel_mis_mean)
            if gate_mis_mean is not None: gate_mis_vals.append(gate_mis_mean)

            name0 = self.test_dataset.load_name(index - 1) if hasattr(self.test_dataset, "load_name") else str(index)
            stats_rows.append({
                "name": name0,
                "psnr": psnr, "ssim": ssim, "l1": l1_loss, "lpips": pl,
                "psnr_mis": psnr_mis, "ssim_mis": ssim_mis, "l1_mis": l1_mis, "lpips_mis": pl_mis,
                "sim": sim_mean, "rel": rel_mean, "gate": gate_mean,
                "sim_mis": sim_mis_mean, "rel_mis": rel_mis_mean, "gate_mis": gate_mis_mean,
            })

            # =============== save images (matched) ===============
            images_joint = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(outputs_img),
                self.postprocess(outputs_merged),
                img_per_row=1
            )

            path_masked = os.path.join(self.results_path, self.model_name, 'masked4060')
            path_result = os.path.join(self.results_path, self.model_name, 'result4060')
            path_joint = os.path.join(self.results_path, self.model_name, 'joint4060')

            name = self.test_dataset.load_name(index - 1)[:-4] + '.png' if hasattr(self.test_dataset,
                                                                                   "load_name") else f"{index:06d}.png"

            create_dir(path_masked)
            create_dir(path_result)
            create_dir(path_joint)

            masked_images = self.postprocess(images * (1 - masks) + masks)[0]
            images_result = self.postprocess(outputs_merged)[0]

            images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
            imsave(masked_images, os.path.join(path_masked, name))
            imsave(images_result, os.path.join(path_result, name))

            print(name + ' complete!')


        csv_path = os.path.join(self.results_path, self.model_name, "reliability_stats.csv")
        try:
            create_dir(os.path.join(self.results_path, self.model_name))
            if len(stats_rows) > 0:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
                    writer.writeheader()
                    for r in stats_rows:
                        writer.writerow(r)
                print("[SAVE] reliability stats to:", csv_path)
        except Exception as e:
            print("[WARN] save csv failed:", e)


        def _bucket_mean(sim_list, gate_list, bins):
            out = []
            if len(sim_list) == 0 or len(gate_list) == 0:
                return out
            sim_arr = np.array(sim_list, dtype=np.float32)
            gate_arr = np.array(gate_list, dtype=np.float32)
            for i in range(len(bins) - 1):
                l, r = float(bins[i]), float(bins[i + 1])
                m = (sim_arr >= l) & (sim_arr < r)
                if m.sum() == 0:
                    out.append((l, r, None, 0))
                else:
                    out.append((l, r, float(gate_arr[m].mean()), int(m.sum())))
            return out

        def _quantile_bins(sim_list, qs=(0.0, 0.33, 0.66, 1.0)):
            if len(sim_list) == 0:
                return [0.0, 0.33, 0.66, 1.01]
            a = np.array(sim_list, dtype=np.float32)
            b = [float(np.quantile(a, q)) for q in qs]
            b[-1] = max(b[-1], 1.0) + 1e-3
            out = [b[0]]
            for x in b[1:]:
                if x <= out[-1] + 1e-6:
                    out.append(out[-1] + 1e-3)
                else:
                    out.append(x)
            return out

        bins_matched = _quantile_bins(sim_vals)
        bins_mismatch = _quantile_bins(sim_mis_vals)

        bucket_matched = _bucket_mean(sim_vals, gate_vals, bins_matched)
        bucket_mismatch = _bucket_mean(sim_mis_vals, gate_mis_vals, bins_mismatch)

        if len(bucket_matched) > 0:
            print("\n[Bucketed gate vs sim] Matched (quantile bins):")
            for l, r, mg, cnt in bucket_matched:
                print(f"  sim in [{l:.4f},{r:.4f}): gate_mean={mg} (n={cnt})")

        if len(bucket_mismatch) > 0:
            print("\n[Bucketed gate vs sim] Mismatch* (quantile bins):")
            for l, r, mg, cnt in bucket_mismatch:
                print(f"  sim in [{l:.4f},{r:.4f}): gate_mean={mg} (n={cnt})")

        def _pearson(x, y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            if len(x) < 3 or len(y) < 3:
                return None
            if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                return None
            return float(np.corrcoef(x, y)[0, 1])

        corr_matched = _pearson(sim_vals, gate_vals)
        corr_mismatch = _pearson(sim_mis_vals, gate_mis_vals)
        print("\n[Correlation] pearson(sim, gate):",
              f"matched={corr_matched}, mismatch*={corr_mismatch}")

        if len(gate_for_drop) == len(psnr_drop_list) and len(gate_for_drop) >= 3:
            corr_gate_drop_psnr = _pearson(gate_for_drop, psnr_drop_list)
            corr_gate_drop_lpips = _pearson(gate_for_drop, lpips_drop_list)
            print("[Correlation] pearson(gate_mis, drop_psnr):", corr_gate_drop_psnr)
            print("[Correlation] pearson(gate_mis, drop_lpips):", corr_gate_drop_lpips)

        print('\nEnd Testing')

        def _avg(x):
            return float(np.average(x)) if len(x) > 0 else float("nan")

        m_psnr, m_ssim, m_l1, m_lpips = _avg(psnr_list), _avg(ssim_list), _avg(l1_list), _avg(lpips_list)
        mm_psnr, mm_ssim, mm_l1, mm_lpips = _avg(psnr_mis_list), _avg(ssim_mis_list), _avg(l1_mis_list), _avg(
            lpips_mis_list)

        print('[Matched Avg]  psnr:{:.3f}  ssim:{:.4f}  l1:{:.5f}  lpips:{:.5f}'.format(
            m_psnr, m_ssim, m_l1, m_lpips
        ))

        if len(psnr_mis_list) > 0:
            print('[Mismatch* Avg] psnr:{:.3f}  ssim:{:.4f}  l1:{:.5f}  lpips:{:.5f}'.format(
                mm_psnr, mm_ssim, mm_l1, mm_lpips
            ))
            print('[Drop: mismatch* - matched] psnr:{:.3f}  ssim:{:.4f}  l1:{:.5f}  lpips:{:.5f}'.format(
                mm_psnr - m_psnr,
                mm_ssim - m_ssim,
                mm_l1 - m_l1,
                mm_lpips - m_lpips
            ))
        else:
            print('[Mismatch] Not computed (no reference available in test set).')
            print('提示：你需要让 test_dataset 返回三元组 (img, mask, ref)，并确保 ref_flist 有配置。')

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, win_size=3, channel_axis=-1, data_range=255)


        return psnr, ssim

    class cal_mean_nme():
        sum = 0
        amount = 0
        mean_nme = 0

        def __call__(self, nme):
            self.sum += nme
            self.amount += 1
            self.mean_nme = self.sum / self.amount
            return self.mean_nme

        def get_mean_nme(self):
            return self.mean_nme
