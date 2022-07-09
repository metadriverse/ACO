import torch
import torch.nn as nn

class LabelMoCoLoss(nn.Module):
    def __init__(self, T, device, channel, thres):
        super().__init__()
        self.T = T
        self.device = device
        self.channel = channel   
        self.thres = thres
        self.CELoss = nn.CrossEntropyLoss().cuda()
        self.LSM = nn.LogSoftmax(dim=1)

    def forward(self, q, k, label, queue_k, queue_label):
        ''' 
        q:           N * C
        k:           N * C
        label:       N
        queue_k:         K * C
        queue_label:     K
        '''
        
        batch_size, total_channel = q.shape
        k_size = queue_k.shape[0]

        assert total_channel % self.channel == 0
        T = total_channel // self.channel 

        # split input features into different categories:
        # N * C -> [N * c, N * c, ...], 
        # C = T * c
        split_list = [self.channel for i in range(T)]
        q_list = torch.split(q, split_list, dim=1)
        # k = torch.cat([k, queue_k], dim=0) # (N+K)*C
        k_list = torch.split(k, split_list, dim=1)
        queue_k_list = torch.split(queue_k, split_list, dim=1)

        # compute logits: [N * (N+K), ...]
        # anchor_dot_contrast_list = [torch.div( 
        #         q_list[i] @ k_list[i].T, self.T)
        #         for i in range(T)]
        anchor_dot_contrast_list = []
        for i in range(2):
            l_self = torch.einsum('nc,nc->n', [q_list[i], k_list[i]]).unsqueeze(1)
            l_other = torch.einsum('nc,kc->nk', [q_list[i], queue_k_list[i]])
            anchor_dot_contrast_list.append(torch.cat([l_self, l_other], dim=1))

        # compute mask, [N * (N+K), ...]
        # 0. first mask, instance mask 
        mask_instance = torch.scatter(
                torch.zeros(batch_size, 1 + k_size).to(self.device),
                1, 
                torch.zeros(batch_size, dtype=torch.int64).view(-1,1).to(self.device),
                1
                )
        # 1. second mask, action label mask
        mask_action_label = (torch.abs(label.unsqueeze(1) - queue_label.unsqueeze(1).T) < self.thres).float().to(self.device)
        mask_action_label = torch.cat([torch.ones(batch_size, 1).to(self.device), mask_action_label], dim=1)

        # compute E(log likelihood)
        loss_list, acc1, acc5 = [], [], []
        for i, anchor_dot_contrast in enumerate(anchor_dot_contrast_list):
            if i == 0:
                mask = mask_instance
            else:
                mask = mask_action_label

            with torch.no_grad():
                # compute top1 acc
                ntop1 = 1 if i == 0 else anchor_dot_contrast.shape[1] // 10 
                _, ind = torch.topk(anchor_dot_contrast, ntop1, 1, True, True)
                ind = torch.scatter(torch.zeros_like(anchor_dot_contrast), 1, ind, 1).to(self.device)
                correct = torch.logical_and(ind, mask)
                acc1.append(correct.sum() / mask.sum())
                # compute top5 acc
                ntop5 = 5 if i == 0 else anchor_dot_contrast.shape[1] // 2
                _, ind = torch.topk(anchor_dot_contrast, ntop5, 1, True, True)
                ind = torch.scatter(torch.zeros_like(anchor_dot_contrast), 1, ind, 1).to(self.device)
                correct = torch.logical_and(ind, mask)
                acc5.append(correct.sum() / mask.sum())

            anchor_dot_contrast /= 0.07
            lsm_logits = self.LSM(anchor_dot_contrast)
            e = (mask * lsm_logits).sum(1) / mask.sum(1)
            loss = (-e).mean()
            # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            # logits = anchor_dot_contrast - logits_max.detach()
            # exp_logits = torch.exp(logits)
            # neg_exp_logits = exp_logits * (1-mask)
            # log_prob = logits - torch.log(neg_exp_logits.sum(1, keepdim=True))
            # expected_log_prob_over_pos = (mask * log_prob).sum(1) / mask.sum(1)
            # loss = -1 * expected_log_prob_over_pos
            # loss = loss.mean()

            # anchor_dot_contrast /= 0.1
            # target = mask / mask.sum(dim=1, keepdim=True)

            # loss = self.CELoss(anchor_dot_contrast, target)

            loss_list.append(loss)

        return loss_list, acc1, acc5

if __name__ == '__main__':
    device = 'cpu'
    loss = LabelMoCoLoss(1, device)
    channel = 10
    bs =  3
    k_size = 5
    q = torch.zeros([bs, channel * 2])
    k = torch.zeros_like(q)
    label = torch.zeros([bs], dtype=torch.int64)
    queue_k = torch.zeros([k_size, channel * 2])
    queue_label = torch.zeros([k_size], dtype=torch.int64)
    res = loss(channel, q, k, label, queue_k, queue_label)
    print(res)

    
        
        

       

