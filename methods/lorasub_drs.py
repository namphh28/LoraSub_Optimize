import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist
from copy import deepcopy

# Assuming these are defined elsewhere; replace with actual implementations if needed
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_lora import SiNet
from models.vit_lora import Attention_LoRA
from utils.schedulers import CosineSchedule
from utils.losses import AugmentedTripletLoss
import optimgrad  # Custom optimizer, assumed to be defined

class LoRAsub_DRS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError(f'Unknown net: {args["net_type"]}.')

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

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info(f'Learning on {self._known_classes}-{self._total_classes}')

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

        # Freeze all parameters initially
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

        # Verify enabled parameters
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
            self.update_optim_transforms()
            self.run_epoch = self.epochs

        self.train_function(train_loader, test_loader)

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
            model.train()
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

                # Debug gradient tracking
                print(f"Epoch {epoch+1}, Batch {i+1}: logits.grad_fn={logits.grad_fn}, features.grad_fn={features.grad_fn}")

                loss = F.cross_entropy(logits, targets)
                ATL = criterion(feature, labels, self._protos)
                loss += self.lambada * ATL

                # Debug loss
                print(f"Loss requires_grad: {loss.requires_grad}, has grad_fn: {loss.grad_fn is not None}")

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
            info = f'Task {self._cur_task}, Epoch {epoch + 1}/{self.run_epoch} => Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}'
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
                # Convert to PyTorch tensor
                self._protos.append(torch.tensor(class_mean, device=self._device, requires_grad=False))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def eval_task(self):
        # Normalize prototypes
        protos = torch.stack(self._protos) if self._protos else torch.tensor([], device=self._device)
        protos = protos / (torch.norm(protos, dim=1, keepdim=True) + self.EPSILON)
        y_pred, y_true = self._eval_model(self.test_loader, protos.cpu().numpy())
        nme_accy = self._evaluate(y_pred.T[0], y_true)
        return nme_accy

    def _eval_model(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T
        dists = cdist(class_means, vectors, 'sqeuclidean')
        scores = dists.T
        return np.argsort(scores, axis=1)[:, :self.topk], y_true

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
                      not bool(re.search('classifier_pool', n)) and p.requires_grad]
        cls_params = [p for n, p in self._network.named_parameters() if bool(re.search('classifier_pool', n))]
        model_optimizer_arg = {
            'params': [
                {'params': fea_params, 'svd': True, 'lr': lr, 'thres': 0.99},
                {'params': cls_params, 'weight_decay': self.weight_decay, 'lr': self.fc_lrate}
            ],
            'weight_decay': self.weight_decay,
            'betas': (0.9, 0.999)
        }
        self.model_optimizer = getattr(optimgrad, self.args['optim'])(**model_optimizer_arg)
        self.model_scheduler = CosineSchedule(self.model_optimizer, K=self.epochs)

    def update_optim_transforms(self):
        if not hasattr(self, 'fisher_prev_list'):
            self.fisher_prev_list = []

        # Store current parameters
        theta_t_minus_1 = {name: param.clone() for name, param in self._network.named_parameters()}

        # Create unconstrained model
        model_unconstrained = deepcopy(self._network).to(self._device)
        for name, param in model_unconstrained.named_parameters():
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

        # Debug enabled parameters
        enabled = set()
        for name, param in model_unconstrained.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"model_unconstrained parameters with requires_grad=True: {enabled}")

        # Create optimizer for unconstrained model
        temp_optimizer = torch.optim.Adam(
            [p for p in model_unconstrained.parameters() if p.requires_grad],
            lr=self.lrate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        temp_scheduler = CosineSchedule(temp_optimizer, K=self.epochs)

        # Train unconstrained model
        model_unconstrained.train()
        for _ in range(self.epochs):
            self.train_function(self.train_loader, None, temp_optimizer, temp_scheduler, model_unconstrained)
        theta_hat_t = {name: param.clone() for name, param in model_unconstrained.named_parameters()}

        # Compute Fisher Information Matrix
        fisher_t = self.compute_fisher_information(self.train_loader, model_unconstrained)

        # Compute lambda_star
        lambda_star = self.compute_lambda_star(theta_hat_t, theta_t_minus_1, fisher_t, self.fisher_prev_list)

        # Adjust LoRA parameters
        with torch.no_grad():
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    for task_id in range(self._cur_task):
                        module.lora_A_k[task_id].weight.data *= (1 - lambda_star)
                        module.lora_A_v[task_id].weight.data *= (1 - lambda_star)
                        module.lora_B_k[task_id].weight.data *= (1 - lambda_star)
                        module.lora_B_v[task_id].weight.data *= (1 - lambda_star)

        # Update optimizer transforms
        self.model_optimizer.get_eigens(self.fea_in)
        self.model_optimizer.get_transforms()
        self.fea_in = defaultdict(dict)

        # Store Fisher matrix
        self.fisher_prev_list.append(fisher_t)

    def compute_lambda_star(self, theta_t, theta_prev, fisher_t, fisher_prev_list):
        delta_theta = {name: theta_t[name] - theta_prev[name] for name in theta_t}
        numerator = sum((delta_theta[name] * fisher_t[name]).sum() for name in fisher_t)
        denominator = sum((delta_theta[name] * (fisher_t[name] + sum(f_prev.get(name, torch.zeros_like(fisher_t[name]).to(self._device)) for f_prev in fisher_prev_list))).sum() for name in fisher_t)
        return numerator / (denominator + self.EPSILON)

    def compute_fisher_information(self, data_loader, model=None):
        if model is None:
            model = self._network
        model.eval()
        fisher = {name: torch.zeros_like(param).to(self._device) for name, param in model.named_parameters() if param.requires_grad}
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