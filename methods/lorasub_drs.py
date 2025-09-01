import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import logging
import numpy as np
from tqdm import tqdm
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_lora import SiNet
from models.vit_lora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
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
        self.topk = 1
        self.class_num = self._network.class_num
        self.debug = False
        self.fea_in = defaultdict(dict)
        self.fisher_dict = {}  # Lưu trữ Fisher Information Matrix cho mỗi tác vụ
        self.lambda_star = None  # Hệ số hợp nhất λ*

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

    def after_task(self):
        self._known_classes = self._total_classes
        # Tính và lưu FIM sau mỗi tác vụ
        self._compute_fisher()

    def _compute_fisher(self):
        """Tính Fisher Information Matrix cho các tham số LoRA của tác vụ hiện tại."""
        self._network.eval()
        fisher = defaultdict(float)
        for i, (_, inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            outputs = self._network(inputs)['logits']
            log_probs = F.log_softmax(outputs, dim=1)
            for n, p in self._network.named_parameters():
                if p.requires_grad and "lora" in n:  # Chỉ tính FIM cho tham số LoRA
                    grad = torch.autograd.grad(log_probs.mean(), p, create_graph=True)[0]
                    fisher[n] += (grad ** 2).sum().item() / len(self.train_loader)
        self.fisher_dict[self._cur_task] = fisher
        self._network.train()

    def _compute_lambda_star(self):
        """Tính λ* dựa trên FIM của tác vụ hiện tại và các tác vụ cũ."""
        if self._cur_task == 0:
            self.lambda_star = 1.0  # Không cần hợp nhất cho tác vụ đầu tiên
            return

        # Lấy FIM của tác vụ hiện tại và các tác vụ cũ
        fisher_current = self.fisher_dict[self._cur_task]
        fisher_old = defaultdict(float)
        for task_id in range(self._cur_task):
            for n, f in self.fisher_dict[task_id].items():
                fisher_old[n] += f / self._cur_task

        # Tính λ* theo công thức trong BECAME (giả định đơn giản hóa)
        lambda_star = 0.0
        total_fisher = 0.0
        for n, f_current in fisher_current.items():
            f_old = fisher_old.get(n, 0.0)
            total_fisher += f_old + f_current
            if f_old + f_current > 0:
                lambda_star += f_old / (f_old + f_current)
        self.lambda_star = lambda_star / len(fisher_current) if total_fisher > 0 else 0.5
        logging.info(f'Computed λ* for task {self._cur_task}: {self.lambda_star}')

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

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
                if f"classifier_pool.{self._network.module.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_A_k.{self._network.module.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_A_v.{self._network.module.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_B_k.{self._network.module.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_B_v.{self._network.module.numtask - 1}." in name:
                    param.requires_grad_(True)
            except:
                if f"classifier_pool.{self._network.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_A_k.{self._network.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_A_v.{self._network.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_B_k.{self._network.numtask - 1}." in name:
                    param.requires_grad_(True)
                if f"lora_B_v.{self._network.numtask - 1}." in name:
                    param.requires_grad_(True)

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
                        self.fea_in[module.lora_A_k[self._cur_task].weight] = deepcopy(module.cur_matrix).to(self._device)
                        self.fea_in[module.lora_A_v[self._cur_task].weight] = deepcopy(module.cur_matrix).to(self._device)
                        self.fea_in[module.lora_B_k[self._cur_task].weight] = deepcopy(module.cur_matrix).to(self._device)
                        self.fea_in[module.lora_B_v[self._cur_task].weight] = deepcopy(module.cur_matrix).to(self._device)
                        module.cur_matrix.zero_()
                        module.matrix_kv = 0
                        module.n_cur_matrix = 0

            self.init_model_optimizer()
            if self._cur_task == 0:
                self.run_epoch = self.init_epoch
            else:
                self._compute_lambda_star()  # Tính λ* trước khi cập nhật optimizer
                self.update_optim_transforms()
                self.run_epoch = self.epochs

        self.train_function(train_loader, test_loader)

    def update_optim_transforms(self):
        """Cập nhật không gian DRS với phép trừ có trọng số λ*."""
        if self._cur_task == 0:
            return

        # Áp dụng phép trừ có trọng số λ* cho các trọng số LoRA cũ
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                for task_id in range(self._cur_task):
                    for param_name in ['lora_A_k', 'lora_A_v', 'lora_B_k', 'lora_B_v']:
                        param = getattr(module, param_name)[self._cur_task].weight
                        old_param = self.fea_in.get(getattr(module, param_name)[task_id].weight, None)
                        if old_param is not None:
                            param.data -= self.lambda_star * old_param  # Trừ có trọng số λ*

        self.model_optimizer.get_eigens(self.fea_in)
        self.model_optimizer.get_transforms()
        self.fea_in = defaultdict(dict)

    def init_model_optimizer(self):
        if self._cur_task == 0:
            lr = self.init_lr
        else:
            lr = self.lrate

        fea_params = [p for n, p in self._network.named_parameters() if
                      not bool(re.search('classifier_pool', n)) and p.requires_grad == True]
        cls_params = [p for n, p in self._network.named_parameters() if bool(re.search('classifier_pool', n))]
        model_optimizer_arg = {
            'params': [
                {'params': fea_params, 'svd': True, 'lr': lr, 'thres': 0.99},
                {'params': cls_params, 'weight_decay': self.weight_decay, 'lr': self.fc_lrate}
            ],
            'weight_decay': self.weight_decay,
            'betas': (0.9, 0.999)
        }
        self.model_optimizer = getattr(optimgrad, self.args['optim'])(**model_optimizer.arg)
        self.model_scheduler = CosineSchedule(self.model_optimizer, K=self.epochs)

    # Các hàm khác (train_function, _build_protos, _evaluate, _eval_model, _compute_accuracy_domain) giữ nguyên
