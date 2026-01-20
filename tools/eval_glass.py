# Copyright 2025 Zujian Dong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup


class GlassMetrics(object):
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.iou_sum = 0
        self.f_beta_sum = 0
        self.mae_sum = 0
        self.ber_sum = 0
        self.count = 0

    def update(self, output, target):
        with torch.no_grad():
            batch_size = output.size(0)
            if output.shape[1] > 1:
                pred_prob = F.softmax(output, dim=1)[:, 1, :, :]
            else:
                pred_prob = torch.sigmoid(output).squeeze(1)
            target = target.float()

            for i in range(batch_size):
                p = pred_prob[i]
                t = target[i]
                mae = torch.abs(p - t).mean()
                p_bin = (p > 0.5).float()
                tp = (p_bin * t).sum()
                tn = ((1 - p_bin) * (1 - t)).sum()
                fp = (p_bin * (1 - t)).sum()
                fn = ((1 - p_bin) * t).sum()
                iou = tp / (tp + fp + fn + 1e-6)
                beta2 = 0.3
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-6)
                tpr = recall
                tnr = tn / (tn + fp + 1e-6)
                ber = 1 - 0.5 * (tpr + tnr)
                self.iou_sum += iou.item()
                self.f_beta_sum += f_beta.item()
                self.mae_sum += mae.item()
                self.ber_sum += ber.item()
                self.count += 1

    def get(self):
        if self.count == 0:
            return 0, 0, 0, 0
        return (self.iou_sum / self.count, 
                self.f_beta_sum / self.count, 
                self.mae_sum / self.count, 
                self.ber_sum / self.count)

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])  

        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='test', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        # val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1, drop_last=False)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.model.to(self.device)
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)
        self.glass_metric = GlassMetrics(self.device)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.metric.reset()
        self.glass_metric.reset()
        self.model.eval()
        
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                output = model.evaluate(image)
            self.metric.update(output, target)
            self.glass_metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            if (i + 1) % 10 == 0:
                logging.info("Sample: {:d}/{:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                    i + 1, len(self.val_loader), pixAcc * 100, mIoU * 100))
        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        g_iou, g_fbeta, g_mae, g_ber = self.glass_metric.get()
        
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))
        print("\n" + "="*40)
        print("Glass Segmentation Metrics (Paper Standard):")
        print("{:<10} {:<10} {:<10} {:<10}".format("IoU", "F-beta", "MAE", "BER"))
        print("{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(g_iou*100, g_fbeta, g_mae, g_ber*100))
        print("="*40 + "\n")
        logging.info('Glass Metrics -> IoU: {:.2f}, F-beta: {:.4f}, MAE: {:.4f}, BER: {:.2f}'.format(
            g_iou * 100, g_fbeta, g_mae, g_ber * 100))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))
if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)
    evaluator = Evaluator(args)
    evaluator.eval()
