import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import _create_model_training_folder
from tqdm import tqdm
from DCL import DCL, DCLW


class Trainer:
    def __init__(self, online_network1, online_network2, target_network1, target_network2, predictor1, predictor2,
                 predictor1_orth, predictor2_orth, optimizer, scheduler, device, **params):
        self.online_network1 = online_network1
        self.online_network2 = online_network2
        self.target_network1 = target_network1
        self.target_network2 = target_network2
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.predictor1 = predictor1
        self.predictor2 = predictor2
        self.predictor1_orth = predictor1_orth
        self.predictor2_orth = predictor2_orth

        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.loss = params['loss']
        self.temp = params['temp']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_1_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network1.parameters(), self.target_network1.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _update_target_network_2_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network2.parameters(), self.target_network2.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    @staticmethod
    def kl_divergence(tensor1, tensor2):
        tensor1 = F.normalize(tensor1, dim=1)
        tensor2 = F.normalize(tensor2, dim=1)
        tensor1_log = F.log_softmax(tensor1, dim=1)
        tensor2_softmax = F.softmax(tensor2, dim=1)
        kl_div = -torch.sum(tensor1_log*tensor2_softmax, dim=1).mean()
        return kl_div

    @staticmethod
    def dot_product_squared(tensor1, tensor2):
        tensor1 = F.normalize(tensor1, dim=1)
        tensor2 = F.normalize(tensor2, dim=1)
        dot_result = torch.einsum("ij,ij->i", tensor1, tensor2)
        squared_result = dot_result ** 2
        return squared_result.mean()

    def initializes_target_1_network(self):
        for param_q, param_k in zip(self.online_network1.parameters(), self.target_network1.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def initializes_target_2_network(self):
        for param_q, param_k in zip(self.online_network2.parameters(), self.target_network2.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_1_network()
        self.initializes_target_2_network()
        iters = self.max_epochs * 177596 // self.batch_size
        for epoch_counter in range(self.max_epochs):
            total_num, total_loss, train_bar = 0, 0, tqdm(train_loader)
            for idx, (batch_md1_view1, batch_md1_view2, batch_md2_view1, batch_md2_view2, _, _) in enumerate(train_bar):
                batch_md1_view1 = batch_md1_view1.to(self.device)
                batch_md1_view2 = batch_md1_view2.to(self.device)
                batch_md2_view1 = batch_md2_view1.to(self.device)
                batch_md2_view2 = batch_md2_view2.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_md1_view1[:32])
                    self.writer.add_image('batch_md1_view1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_md1_view2[:32])
                    self.writer.add_image('batch_md1_view2', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_md2_view1[:32])
                    self.writer.add_image('batch_md2_view1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_md2_view2[:32])
                    self.writer.add_image('batch_md2_view2', grid, global_step=niter)

                loss_1, out_1, out_1_m = self.update1(batch_md1_view1, batch_md1_view2)
                loss_2, out_2, out_2_m = self.update2(batch_md2_view1, batch_md2_view2)
                self.writer.add_scalar('inner model loss', loss_1+loss_2, global_step=niter)

                if self.loss == 'dcl':
                    l = DCL(temperature=self.temp)
                    loss_3 = l(out_1, out_2) + l(out_2, out_1)
                elif self.loss == 'dclw':
                    l = DCLW(temperature=self.temp)
                    loss_3 = l(out_1, out_2) + l(out_2, out_1)
                elif self.loss == 'ce':
                    out = torch.cat([out_1, out_2], dim=0)
                    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / 0.1)
                    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size,
                                                                    device=sim_matrix.device)).bool()
                    sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

                    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / 0.1)

                    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                    loss_3 = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

                sim_m2p = out_1 @ out_2.t() / self.temp
                sim_m2p_m = out_1_m @ out_2.t() / self.temp
                loss_KL_sim_m2p = -torch.sum(F.log_softmax(sim_m2p, dim=1)*F.softmax(sim_m2p_m, dim=1),dim=1).mean()
                sim_p2m = out_2 @ out_1.t() / self.temp
                sim_p2m_m = out_2_m @ out_1.t() / self.temp
                loss_KL_sim_p2m = -torch.sum(F.log_softmax(sim_p2m, dim=1) * F.softmax(sim_p2m_m, dim=1), dim=1).mean()
                a = 0.4
                loss_pmc = (a*(loss_KL_sim_m2p+loss_KL_sim_p2m)+(1-a)*loss_3)/2
                self.writer.add_scalar('inter model loss', loss_pmc, global_step=niter)
                b = 0.4
                loss = b*(loss_1 + loss_2) + (1-b)*loss_pmc
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(self.max_epochs + idx / iters)
                self._update_target_network_1_parameters()
                self._update_target_network_2_parameters()

                niter += 1
                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                train_bar.set_description(
                    'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch_counter, self.max_epochs, total_loss / total_num))
                self.writer.add_scalar('total loss', total_loss / total_num, global_step=niter)
            print("End of epoch {}".format(epoch_counter))

        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update1(self, batch_view_1, batch_view_2):
        online_from_view_1, orth_online_from_view_1 = self.online_network1(batch_view_1)
        predictions_from_view_1 = self.predictor1(online_from_view_1)
        orth_predictions_from_view_1 = self.predictor1_orth(orth_online_from_view_1)
        online_from_view_2, orth_online_from_view_2 = self.online_network1(batch_view_2)
        predictions_from_view_2 = self.predictor1(online_from_view_2)
        orth_predictions_from_view_2 = self.predictor1_orth(orth_online_from_view_2)

        with torch.no_grad():
            targets_to_view_2, orth_targets_to_view2 = self.target_network1(batch_view_1)
            targets_to_view_1, orth_targets_to_view1 = self.target_network1(batch_view_2)


        sim_n2n = predictions_from_view_1 @ predictions_from_view_1.t() / self.temp
        sim_n2n_m = targets_to_view_1 @ targets_to_view_1.t() / self.temp
        sim_loss_KL = -torch.sum(F.log_softmax(sim_n2n, dim=1)*F.softmax(sim_n2n_m, dim=1),dim=1).mean()

        sim_o2o = orth_predictions_from_view_1 @ orth_predictions_from_view_1.t() / self.temp
        sim_o2o_m = orth_targets_to_view1 @ orth_targets_to_view1.t() / self.temp
        sim_loss_KL += -torch.sum(F.log_softmax(sim_o2o, dim=1) * F.softmax(sim_o2o_m, dim=1), dim=1).mean()

        sim_n2n = predictions_from_view_2 @ predictions_from_view_2.t() / self.temp
        sim_n2n_m = targets_to_view_2 @ targets_to_view_2.t() / self.temp
        sim_loss_KL += -torch.sum(F.log_softmax(sim_n2n, dim=1) * F.softmax(sim_n2n_m, dim=1), dim=1).mean()

        sim_o2o = orth_predictions_from_view_2 @ orth_predictions_from_view_2.t() / self.temp
        sim_o2o_m = orth_targets_to_view2 @ orth_targets_to_view2.t() / self.temp
        sim_loss_KL += -torch.sum(F.log_softmax(sim_o2o, dim=1) * F.softmax(sim_o2o_m, dim=1), dim=1).mean()

        l = DCL(temperature=self.temp)
        loss_DC_nomal = 1/2*l(online_from_view_1, online_from_view_2) + 1/2*l(online_from_view_2,online_from_view_1)
        loss_DC_orth = 1/2*l(orth_online_from_view_1, orth_online_from_view_2) + 1/2*l(orth_online_from_view_2, orth_online_from_view_1)
        loss_orth = self.dot_product_squared(online_from_view_1, orth_online_from_view_1)
        a = 0.4
        loss = a*sim_loss_KL + (1-a)*(loss_DC_nomal +loss_DC_orth) + loss_orth
        return loss.mean(), online_from_view_1, targets_to_view_2

    def update2(self, batch_view_1, batch_view_2):
        online_from_view_1, orth_online_from_view_1 = self.online_network2(batch_view_1)
        predictions_from_view_1 = self.predictor1(online_from_view_1)
        orth_predictions_from_view_1 = self.predictor1_orth(orth_online_from_view_1)

        online_from_view_2, orth_online_from_view_2 = self.online_network2(batch_view_2)
        predictions_from_view_2 = self.predictor1(online_from_view_2)
        orth_predictions_from_view_2 = self.predictor1_orth(orth_online_from_view_2)

        with torch.no_grad():
            targets_to_view_2, orth_targets_to_view2 = self.target_network2(batch_view_1)
            targets_to_view_1, orth_targets_to_view1 = self.target_network2(batch_view_2)

        sim_n2n = predictions_from_view_1 @ predictions_from_view_1.t() / self.temp
        sim_n2n_m = targets_to_view_1 @ targets_to_view_1.t() / self.temp
        sim_loss_KL = -torch.sum(F.log_softmax(sim_n2n, dim=1)*F.softmax(sim_n2n_m, dim=1),dim=1).mean()
        sim_o2o = orth_predictions_from_view_1 @ orth_predictions_from_view_1.t() / self.temp
        sim_o2o_m = orth_targets_to_view1 @ orth_targets_to_view1.t() / self.temp
        sim_loss_KL += -torch.sum(F.log_softmax(sim_o2o, dim=1) * F.softmax(sim_o2o_m, dim=1), dim=1).mean()

        sim_n2n = predictions_from_view_2 @ predictions_from_view_2.t() / self.temp
        sim_n2n_m = targets_to_view_2 @ targets_to_view_2.t() / self.temp
        sim_loss_KL += -torch.sum(F.log_softmax(sim_n2n, dim=1) * F.softmax(sim_n2n_m, dim=1), dim=1).mean()
        sim_o2o = orth_predictions_from_view_2 @ orth_predictions_from_view_2.t() / self.temp
        sim_o2o_m = orth_targets_to_view2 @ orth_targets_to_view2.t() / self.temp
        sim_loss_KL += -torch.sum(F.log_softmax(sim_o2o, dim=1) * F.softmax(sim_o2o_m, dim=1), dim=1).mean()

        l = DCL(temperature=self.temp)
        loss_DC_nomal = 1/2*l(online_from_view_1, online_from_view_2) + 1/2*l(online_from_view_2,online_from_view_1)
        loss_DC_orth = 1/2*l(orth_online_from_view_1, orth_online_from_view_2) + 1/2*l(orth_online_from_view_2, orth_online_from_view_1)
        loss_orth = self.dot_product_squared(online_from_view_1, orth_online_from_view_1)
        a = 0.4
        loss = a*sim_loss_KL + (1-a)*(loss_DC_nomal +loss_DC_orth) + loss_orth
        return loss.mean(), online_from_view_1, targets_to_view_2


    def save_model(self, PATH):

        torch.save({
            'online_network_1_state_dict': self.online_network1.state_dict(),
            'target_network_1_state_dict': self.target_network1.state_dict(),
            'online_network_2_state_dict': self.online_network2.state_dict(),
            'target_network1_2_state_dict': self.target_network2.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

