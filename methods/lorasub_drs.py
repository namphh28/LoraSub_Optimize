import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_lora import SiNet
from models.vit_lora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import optimgrad
import re
from collections import defaultdict
from utils.losses import AugmentedTripletLoss
from scipy.spatial.distance import cdist




class LoRAsub_DRS(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))

        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lambada = args["lambada"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.fc_lrate = args["fc_lrate"]
        self.margin_inter = args["margin_inter"]
        self.eval = args['eval']
        self._protos = []

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False
        self.fea_in = defaultdict(dict)

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

    def after_task(self):
        self._known_classes = self._total_classes
        # logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if not self.eval:
            self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._build_protos()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        with torch.no_grad():
            if self._cur_task > 0:
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    self._network(inputs, get_cur_x=True)

                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        self.fea_in[module.lora_A_k[self._cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self._device)
                        self.fea_in[module.lora_A_v[self._cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self._device)
                        self.fea_in[module.lora_B_k[self._cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self._device)
                        self.fea_in[module.lora_B_v[self._cur_task].weight] = deepcopy(module.cur_matrix).to(
                            self._device)
                        module.cur_matrix.zero_()
                        module.matrix_kv = 0
                        module.n_cur_matrix = 0

            self.init_model_optimizer()
            if self._cur_task == 0:
                self.run_epoch = self.init_epoch
            else:
                self.update_optim_transforms()
                self.run_epoch = self.epochs

        self.train_function(train_loader, test_loader)

        return

    def train_function(self, train_loader, test_loader, optimizer=None, scheduler=None, model=None):
        if model is None:
            model = self._network
        if optimizer is None:
            optimizer = self.model_optimizer
        if scheduler is None:
            scheduler = self.model_scheduler
        
        prog_bar = tqdm(range(self.run_epoch))
        criterion = AugmentedTripletLoss(margin=self.margin_inter).to(self._device)
        for _, epoch in enumerate(prog_bar):
            model.train()  # Đặt mô hình ở chế độ huấn luyện
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                ret = model(inputs)
                logits = ret['logits']
                features = ret['features']
                feature = features / features.norm(dim=-1, keepdim=True)
                loss = F.cross_entropy(logits, targets)
                ATL = criterion(feature, labels, self._protos)
                loss += self.lambada * ATL

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10:
                    break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)



    def _build_protos(self):
        self._network.to(self._device)
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)

                self._protos.append(class_mean)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_model(self.test_loader, self._protos / np.linalg.norm(self._protos, axis=1)[:, None])
        nme_accy = self._evaluate(y_pred.T[0], y_true)
        return nme_accy

    def _eval_model(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T

        dists = cdist(class_means, vectors, 'sqeuclidean')
        scores = dists.T

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def init_model_optimizer(self):
        if self._cur_task == 0:
            lr = self.init_lr
        else:
            lr = self.lrate

        fea_params = [p for n, p in self._network.named_parameters() if
                      not bool(re.search('classifier_pool', n)) and p.requires_grad == True]

        cls_params = [p for n, p in self._network.named_parameters() if bool(re.search('classifier_pool', n))]
        model_optimizer_arg = {'params': [{'params': fea_params, 'svd': True, 'lr': lr,
                                           'thres': 0.99},
                                          {'params': cls_params, 'weight_decay': self.weight_decay,
                                           'lr': self.fc_lrate}],
                               'weight_decay': self.weight_decay,
                               'betas': (0.9, 0.999)
                               }
        # self.args['model_optimizer'] = 'Adam'
        self.model_optimizer = getattr(
            optimgrad, self.args['optim'])(**model_optimizer_arg)
        self.model_scheduler = CosineSchedule(self.model_optimizer, K=self.epochs)

    def update_optim_transforms(self):
        # Lưu trữ FIM của các task cũ
        if not hasattr(self, 'fisher_prev_list'):
            self.fisher_prev_list = []

        # Lấy tham số hiện tại (θ_{t-1})
        theta_t_minus_1 = {name: param.clone() for name, param in self._network.named_parameters()}
        
        # Tạo bản sao mô hình và bật requires_grad cho các tham số cần thiết
        model_unconstrained = deepcopy(self._network).to(self._device)
        for name, param in model_unconstrained.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.module.numtask - 1) + "." in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_A_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) + "." in name:
                    param.requires_grad_(True)

        # Tạo optimizer cho model_unconstrained
        temp_optimizer = torch.optim.Adam(
            [p for p in model_unconstrained.parameters() if p.requires_grad],
            lr=self.lrate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        temp_scheduler = CosineSchedule(temp_optimizer, K=self.epochs)
        
        # Huấn luyện không ràng buộc để lấy \hat{θ}_t
        model_unconstrained.train()  # Đặt mô hình ở chế độ huấn luyện
        for _ in range(self.epochs):  # Có thể giảm số epoch để tối ưu
            self.train_function(self.train_loader, None, temp_optimizer, temp_scheduler, model_unconstrained)
        theta_hat_t = {name: param.clone() for name, param in model_unconstrained.named_parameters()}
        
        # Tính FIM cho task hiện tại
        fisher_t = self.compute_fisher_information(self.train_loader, model_unconstrained)
        
        # Tính λ*
        lambda_star = self.compute_lambda_star(theta_hat_t, theta_t_minus_1, fisher_t, self.fisher_prev_list)
        
        # Điều chỉnh tham số LoRA của các task cũ
        with torch.no_grad():
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    for task_id in range(self._cur_task):
                        # Nhân tham số LoRA với (1 - λ*) thay vì trừ toàn bộ
                        module.lora_A_k[task_id].weight.data *= (1 - lambda_star)
                        module.lora_A_v[task_id].weight.data *= (1 - lambda_star)
                        module.lora_B_k[task_id].weight.data *= (1 - lambda_star)
                        module.lora_B_v[task_id].weight.data *= (1 - lambda_star)
        
        # Gọi hàm gốc để cập nhật transforms
        self.model_optimizer.get_eigens(self.fea_in)
        self.model_optimizer.get_transforms()
        self.fea_in = defaultdict(dict)
        
        # Lưu FIM cho task hiện tại
        self.fisher_prev_list.append(fisher_t)

    def compute_lambda_star(self, theta_t, theta_prev, fisher_t, fisher_prev_list):
      """Tính hệ số λ* dựa trên BECAME."""
      delta_theta = {name: theta_t[name] - theta_prev[name] for name in theta_t}
      numerator = sum((delta_theta[name] * fisher_t[name]).sum() for name in fisher_t)
      denominator = sum((delta_theta[name] * (fisher_t[name] + sum(f_prev.get(name, torch.zeros_like(fisher_t[name]).to(self._device)) for f_prev in fisher_prev_list))).sum() for name in fisher_t)
      return numerator / (denominator + self.EPSILON)  # Tránh chia cho 0

    def compute_fisher_information(self, data_loader, model=None):
        """Tính Fisher Information Matrix (diagonal approximation)."""
        if model is None:
            model = self._network
        model.eval()
        fisher = {name: torch.zeros_like(param).to(self._device) for name, param in model.named_parameters()}
        for _, inputs, targets in data_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model.zero_grad()
            outputs = model(inputs)['logits']
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad ** 2) / len(data_loader)
        return fisher

