
class SparseEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, shared_ratio=0.3, alpha=1.0, epsilon=20):
        super(SparseEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shared_ratio = shared_ratio
        self.alpha = alpha
        self.epsilon = epsilon
        self.bn = nn.BatchNorm1d(output_dim)
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        nn.init.orthogonal_(self.weight, gain=1.0)
        nn.init.zeros_(self.bias)
        init_mask_scores,self.prob = self._create_erdos_renyi_mask_scores()
        self.mask_scores = nn.Parameter(init_mask_scores)

    def _create_erdos_renyi_mask_scores(self):
        device = self.weight.device
        prob = (self.epsilon * (self.input_dim + self.output_dim)) / (self.input_dim * self.output_dim)
        mask_scores = torch.full((self.output_dim, self.input_dim), prob, device=device)
        noise = torch.randn(self.output_dim, self.input_dim, device=device) * 0.1
        mask_scores = mask_scores + noise
        mask_scores = torch.clamp(mask_scores, 0.01, 0.99)
        return mask_scores, prob

    def get_mask_scores(self):
        num_elements = self.mask_scores.numel()  
        k = int(self.shared_ratio * num_elements)
        _, indices = torch.topk(self.mask_scores.view(-1), k)
        sparse_mask = torch.zeros(num_elements, dtype=torch.float, device=self.mask_scores.device)
        sparse_mask[indices] = True
        sparse_mask = sparse_mask.view_as(self.mask_scores)
        return sparse_mask

    def forward(self, x, shared_weight):
        sparse_mask = self.get_mask_scores()
        sparse_mask = sparse_mask.to(shared_weight.device)
        combined_weight = sparse_mask * (shared_weight * self.alpha) +  (1.0 - sparse_mask) * self.weight
        if self.training:
            keep_prob = self.prob + (1 - self.prob) * sparse_mask
            dropout_mask = (torch.rand_like(combined_weight) < keep_prob).float()
            combined_weight = combined_weight * dropout_mask / (keep_prob + 1e-8)
        out = F.linear(x, combined_weight, self.bias)
        out = self.bn(out)
        return out