# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
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
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
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

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

class LabelMoCo(MoCo):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, ckpt=None):
        nn.Module.__init__(self)
        
        self.K = K
        self.m = m
        self.T = T

        self.register_buffer("label", torch.zeros([K], dtype=torch.float32))
        self.label = torch.sub(self.label, 1)

        self.register_buffer("queue", torch.randn(K, dim * 2))
        self.queue[:, :dim] = nn.functional.normalize(self.queue[:, :dim], dim=1)
        self.queue[:, dim:] = nn.functional.normalize(self.queue[:, dim:], dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.encoder_q = base_encoder(num_classes=dim * 2)

        self.ckpt = ckpt
        if ckpt is not None:
            ckpt_dict = torch.load(ckpt)['state_dict']
            from collections import OrderedDict
            d = OrderedDict()
            prefix = 'module.encoder_q.'
            loaded_part = ['conv1', 'bn1', 'layer1', 'layer2']
            for k,v in ckpt_dict.items():
                for lp in loaded_part:
                    if prefix+lp in k:
                        d[k[17:]] = v
            self.encoder_q.load_state_dict(d, strict=False)
                        
            for p in self.encoder_q.conv1.parameters():
                p.requires_grad = False
            for p in self.encoder_q.layer1.parameters():
                p.requires_grad = False
            for p in self.encoder_q.layer2.parameters():
                p.requires_grad = False
            self.encoder_q.bn1.weight.requires_grad = False          
            self.encoder_q.bn1.bias.requires_grad = False          

        self.encoder_k = base_encoder(num_classes=dim * 2)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    def set_train_mode(self):
        self.train()
        # if self.ckpt is not None:
        #     self.encoder_q.conv1.eval()
        #     self.encoder_q.bn1.eval()
        #     self.encoder_q.layer1.eval()
        #     self.encoder_q.layer2.eval()
        #     self.encoder_k.conv1.eval()
        #     self.encoder_k.bn1.eval()
        #     self.encoder_k.layer1.eval()
        #     self.encoder_k.layer2.eval()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[ptr:ptr+batch_size, :] = keys
        self.label[ptr:ptr+batch_size] = labels
        ptr = (ptr+batch_size) % self.K

        self.queue_ptr[0] = ptr
    
    def extract_feature(self, img):
        feat = self.encoder_q(img)
        dim = feat.shape[1] // 2
        feat1 = nn.functional.normalize(feat[:, :dim], dim=1)
        feat2 = nn.functional.normalize(feat[:, dim:], dim=1)

        return feat1, feat2

    def forward(self, im_q, im_k, label, is_eval=False):
        q = self.encoder_q(im_q)
        dim = q.shape[1] // 2

        q1 = nn.functional.normalize(q[:, :dim], dim=1)
        q2 = nn.functional.normalize(q[:, dim:], dim=1)
        q = torch.cat([q1, q2], dim=1)

        with torch.no_grad():  # no gradient to keys
            if not is_eval:
                self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC

            k1 = nn.functional.normalize(k[:, :dim], dim=1)
            k2 = nn.functional.normalize(k[:, dim:], dim=1)
            k = torch.cat([k1, k2], dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        queue_k = self.queue.clone().detach()
        queue_label = self.label.clone()

        if not is_eval:
            self._dequeue_and_enqueue(k, label)

        return q, k, queue_k, queue_label

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
