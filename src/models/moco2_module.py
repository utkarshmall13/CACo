from argparse import ArgumentParser
from itertools import chain
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from pl_bolts.metrics import precision_at_k

from datasets.seco_datamodule import ChangeAwareContrastMultiAugDataModule, \
    TemporalContrastMultiAugDataModule, \
    SeasonalContrastMultiAugDataModule, \
    SeasonalContrastBasicDataModule

MINUSINF = -100000000


class MocoV2(LightningModule):

    def __init__(self, base_encoder, emb_dim, num_negatives, emb_spaces=1, datamodule=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.datamodule = datamodule

        # create the encoders
        template_model = getattr(torchvision.models, base_encoder)
        self.encoder_q = template_model(num_classes=self.hparams.emb_dim)
        self.encoder_k = template_model(num_classes=self.hparams.emb_dim)

        # remove fc layer
        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-1], nn.Flatten())
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-1], nn.Flatten())

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the projection heads
        self.mlp_dim = 512 * (1 if base_encoder in ['resnet18', 'resnet34'] else 4)
        self.heads_q = nn.ModuleList([
            nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, emb_dim))
            for _ in range(emb_spaces)
        ])
        self.heads_k = nn.ModuleList([
            nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, emb_dim))
            for _ in range(emb_spaces)
        ])

        for param_q, param_k in zip(self.heads_q.parameters(), self.heads_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_spaces, emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(emb_spaces, 1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)
        for param_q, param_k in zip(self.heads_q.parameters(), self.heads_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_idx, opt_inds=None):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[queue_idx])
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[queue_idx, :, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[queue_idx] = ptr

    def forward(self, img_q, img_k, mmweights=None, inds=None):
        """
        Input:
            img_q: a batch of query images
            img_k: a batch of key images
        Output:
            logits, targets
        """

        # update the key encoder
        self._momentum_update_key_encoder()

        # compute query features
        v_q = self.encoder_q(img_q)

        # compute key features
        v_k = []
        for i in range(self.hparams.emb_spaces):
            # shuffle for making use of BN
            if self.use_ddp or self.use_ddp2:
                img_k[i], idx_unshuffle = batch_shuffle_ddp(img_k[i])

            with torch.no_grad():  # no gradient to keys
                v_k.append(self.encoder_k(img_k[i]))

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                v_k[i] = batch_unshuffle_ddp(v_k[i], idx_unshuffle)

        logits = []
        for i in range(self.hparams.emb_spaces):
            # compute query projections
            z_q = self.heads_q[i](v_q)  # queries: NxC
            z_q = nn.functional.normalize(z_q, dim=1)

            # compute key projections
            z_k = []
            for j in range(self.hparams.emb_spaces):
                with torch.no_grad():  # no gradient to keys
                    z_k.append(self.heads_k[i](v_k[j]))  # keys: NxC
                    z_k[j] = nn.functional.normalize(z_k[j], dim=1)

            if i==1:  # temporally invariant
                if isinstance(self.datamodule, ChangeAwareContrastMultiAugDataModule):
                    with torch.no_grad():
                        l_q = nn.functional.normalize(self.heads_k[1](self.encoder_k(img_q)), dim=1)
                        l_k = nn.functional.normalize(self.heads_k[1](self.encoder_k(img_k[1])), dim=1)

                        zlt = l_q.view(-1, 2, l_q.size(1))
                        zlt = torch.flip(zlt, (1, ))
                        zlt = zlt.view(l_q.size(0), l_q.size(1))

                        diffsst = torch.sqrt(torch.sum((l_q-l_k)*(l_q-l_k), dim=1)).cpu().numpy()
                        diffslt = torch.sqrt(torch.sum((l_q-zlt)*(l_q-zlt), dim=1)).cpu().numpy()
                        epsilon = 0.0001
                        diffs = diffslt/(diffsst+epsilon)

                        momentum=True
                        factor=4

                        if momentum:
                            for ind, diffst, difflt, diff in zip(inds, diffsst, diffslt, diffs):
                                num = self.datamodule.train_dataset.loss[ind][:]
                                den = self.datamodule.train_dataset.lossdeno[ind]
                                self.datamodule.train_dataset.loss[ind] = ((num*factor)+np.array([difflt, diffst, diff]))/((den*factor)+1)
                                self.datamodule.train_dataset.lossdeno[ind]=1
                        else:
                            for ind, diffst, difflt, diff in zip(inds, diffsst, diffslt, diffs):
                                self.datamodule.train_dataset.loss[ind][:] = np.array([difflt, diffst, diff])
                                self.datamodule.train_dataset.lossdeno[ind]=1

            if isinstance(self.datamodule, SeasonalContrastMultiAugDataModule) or isinstance(self.datamodule, SeasonalContrastBasicDataModule):
                # select positive and negative pairs
                z_pos = z_k[i]
                z_neg = self.queue[i].clone().detach()
                if i > 0:  # embedding space 0 is invariant to all augmentations
                    z_neg = torch.cat([z_neg, *[z_k[j].T for j in range(self.hparams.emb_spaces) if j != i]], dim=1)

                # compute logits
                # Einstein sum is more intuitive
                l_pos = torch.einsum('nc,nc->n', z_q, z_pos).unsqueeze(-1)  # positive logits: Nx1
                l_neg = torch.einsum('nc,ck->nk', z_q, z_neg)  # negative logits: NxK

            elif isinstance(self.datamodule, TemporalContrastMultiAugDataModule):
                # select positive and negative pairs
                z_pos_0 = z_k[i][0::2]
                z_pos_1 = z_k[i][1::2]

                z_q_0 = z_q[0::2]
                z_q_1 = z_q[1::2]
                z_q = torch.cat([z_q_0, z_q_1], dim=0)

                z_neg_0 = self.queue[i].clone().detach()[:, 0::2]
                z_neg_1 = self.queue[i].clone().detach()[:, 1::2]

                if i==0:
                    z_neg_0 = torch.cat([z_neg_0, z_k[0][1::2].T], dim=1)
                    z_neg_1 = torch.cat([z_neg_1, z_k[0][0::2].T], dim=1)
                if i==1:
                    z_neg_0 = torch.cat([z_neg_0, z_k[1][1::2].T, z_k[0].T, z_k[2].T], dim=1)
                    z_neg_1 = torch.cat([z_neg_1, z_k[1][0::2].T, z_k[0].T, z_k[2].T], dim=1)
                if i==2:
                    z_neg_0 = torch.cat([z_neg_0, z_k[2][1::2].T, z_k[0].T, z_k[1].T], dim=1)
                    z_neg_1 = torch.cat([z_neg_1, z_k[2][0::2].T, z_k[0].T, z_k[1].T], dim=1)
                # print(z_pos_0.size(), z_pos_1.size(), z_neg_0.size(), z_neg_1.size())

                teco_new = True
                if teco_new:
                    # compute logits
                    # Einstein sum is more intuitive
                    l_pos_0 = torch.einsum('nc,nc->n', z_q_0, z_pos_0).unsqueeze(-1)  # positive logits: N/2x1
                    l_pos_1 = torch.einsum('nc,nc->n', z_q_1, z_pos_1).unsqueeze(-1)  # positive logits: N/2x1
                    l_neg_0 = torch.einsum('nc,ck->nk', z_q_0, z_neg_0)  # negative logits: N/2xK/2
                    l_neg_1 = torch.einsum('nc,ck->nk', z_q_1, z_neg_1)  # negative logits: N/2xK/2

                    l_pos = torch.cat([l_pos_0, l_pos_1], dim=0)
                    l_neg = torch.cat([l_neg_0, l_neg_1], dim=0)

                else:
                    z_pos = torch.cat([z_pos_0, z_pos_1], dim=0)
                    z_neg = torch.cat([z_neg_0, z_neg_1], dim=1)

                    # compute logits
                    # Einstein sum is more intuitive
                    l_pos = torch.einsum('nc,nc->n', z_q, z_pos).unsqueeze(-1)  # positive logits: Nx1
                    l_neg = torch.einsum('nc,ck->nk', z_q, z_neg)  # negative logits: NxK

            elif isinstance(self.datamodule, ChangeAwareContrastMultiAugDataModule):
                # select positive and negative pairs
                z_pos_0 = z_k[i][0::2]
                z_pos_1 = z_k[i][1::2]
         
                mmweight_0 = mmweights[0::2]
                mmweight_1 = mmweights[1::2]

                z_q_0 = z_q[0::2]
                z_q_1 = z_q[1::2]
                z_q = torch.cat([z_q_0, z_q_1], dim=0)

                z_neg_0 = self.queue[i].clone().detach()[:, 0::2]
                z_neg_1 = self.queue[i].clone().detach()[:, 1::2]

                if i==0:
                    z_neg_0 = torch.cat([z_neg_0, z_k[0][1::2].T], dim=1)
                    z_neg_1 = torch.cat([z_neg_1, z_k[0][0::2].T], dim=1)
                if i==1:
                    z_neg_0 = torch.cat([z_neg_0, z_k[0].T, z_k[2].T, z_k[1][1::2].T], dim=1)
                    z_neg_1 = torch.cat([z_neg_1, z_k[0].T, z_k[2].T, z_k[1][0::2].T], dim=1)
                if i==2:
                    z_neg_0 = torch.cat([z_neg_0, z_k[0].T, z_k[1].T, z_k[2][1::2].T], dim=1)
                    z_neg_1 = torch.cat([z_neg_1, z_k[0].T, z_k[1].T, z_k[2][0::2].T], dim=1)
                # print(z_pos_0.size(), z_pos_1.size(), z_neg_0.size(), z_neg_1.size())

                wt_matrix_0 = np.zeros((z_pos_0.size(0), z_pos_0.size(0)))
                wt_matrix_1 = np.zeros((z_pos_1.size(0), z_pos_1.size(0)))
                for ind, mmweight in enumerate(mmweight_0):
                    if mmweight==0:
                        wt_matrix_0[ind, ind] = MINUSINF
                for ind, mmweight in enumerate(mmweight_1):
                    if mmweight==0:
                        wt_matrix_1[ind, ind] = MINUSINF

                wt_matrix_0 = np.concatenate([np.zeros((z_pos_0.size(0), z_neg_0.size(1)-z_pos_0.size(0))), wt_matrix_0], axis=1)
                wt_matrix_1 = np.concatenate([np.zeros((z_pos_1.size(0), z_neg_1.size(1)-z_pos_1.size(0))), wt_matrix_1], axis=1)

                wt_matrix_0 = torch.from_numpy(wt_matrix_0).to(z_pos_0.device)
                wt_matrix_1 = torch.from_numpy(wt_matrix_1).to(z_pos_1.device)

                # compute logits
                # Einstein sum is more intuitive
                l_pos_0 = torch.einsum('nc,nc->n', z_q_0, z_pos_0).unsqueeze(-1)  # positive logits: N/2x1
                l_pos_1 = torch.einsum('nc,nc->n', z_q_1, z_pos_1).unsqueeze(-1)  # positive logits: N/2x1
                l_neg_0 = torch.einsum('nc,ck->nk', z_q_0, z_neg_0)  # negative logits: N/2xK/2
                l_neg_1 = torch.einsum('nc,ck->nk', z_q_1, z_neg_1)  # negative logits: N/2xK/2

                l_neg_0 = l_neg_0+wt_matrix_0
                l_neg_1 = l_neg_1+wt_matrix_1

                l_pos = torch.cat([l_pos_0, l_pos_1], dim=0)
                l_neg = torch.cat([l_neg_0, l_neg_1], dim=0)

            lcat = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
            lcat /= self.hparams.softmax_temperature  # apply temperature
            logits.append(lcat)

            # dequeue and enqueue
            self._dequeue_and_enqueue(z_k[i], queue_idx=i)

        # targets: positive key indicators
        targets = torch.zeros(logits[0].shape[0], dtype=torch.long)
        targets = targets.type_as(logits[0])
        return logits, targets

    def training_step(self, batch, batch_idx):
        # print(self.datamodule)
        if isinstance(self.datamodule, SeasonalContrastBasicDataModule):
            img_q, img_k= batch
        if isinstance(self.datamodule, SeasonalContrastMultiAugDataModule):
            inds, img_q, img_k = batch
        if isinstance(self.datamodule, TemporalContrastMultiAugDataModule):
            inds, img_q, img_k = batch
        if isinstance(self.datamodule, ChangeAwareContrastMultiAugDataModule):
            inds, img_q, img_k, mmweights = batch

        if self.hparams.emb_spaces == 1 and isinstance(img_k, torch.Tensor):
            img_k = [img_k]

        # print(inds)

        if isinstance(self.datamodule, ChangeAwareContrastMultiAugDataModule):
            output, target = self(img_q, img_k, mmweights=mmweights, inds=inds)
        else:
            output, target = self(img_q, img_k)

        losses = []
        accuracies = []
        for out in output:
            losses.append(F.cross_entropy(out.float(), target.long()))
            accuracies.append(precision_at_k(out, target, top_k=(1,))[0])
        loss = torch.sum(torch.stack(losses))

        log = {'train_loss': loss}
        for i, acc in enumerate(accuracies):
            log[f'train_acc/subspace{i}'] = acc

        self.log_dict(log, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = chain(self.encoder_q.parameters(), self.heads_q.parameters())
        optimizer = optim.SGD(params, self.hparams.learning_rate,
                              momentum=self.hparams.momentum,
                              weight_decay=self.hparams.weight_decay)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--base_encoder', type=str, default='resnet18')
        parser.add_argument('--emb_dim', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=20)
        parser.add_argument('--num_negatives', type=int, default=16384)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=256)
        return parser


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_ddp(x):  # pragma: no-cover
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):  # pragma: no-cover
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]
