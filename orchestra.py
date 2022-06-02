# Sinkhorn Knopp 
def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations. 
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments


# Projector 
class projection_MLP_orchestra(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(projection_MLP_orchestra, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x))) 
        x = self.layer2_bn(self.layer2(x)) 
        return x

# Model class
class OrchestraResNet50(nn.Module):
    def __init__(self, pretrained, out_dim, args):
        super(OrchestraResNet50, self).__init__()

        # Setup arguments
        self.N_local = args.num_local_clusters # Number of local clusters
        self.N_centroids = args.num_global_centroids # Number of centroids 
        self.m_size = args.cluster_m_size # Memory size for projector representations 
        self.T = args.temperature

        self.ema_value = args.ema_value

        ### Online Model
        self.backbone = models.resnet50(pretrained=pretrained)
        self.projector = projection_MLP_orchestra(in_dim=self.backbone.fc.out_features, out_dim=out_dim)

        ### Target model
        self.target_backbone = models.resnet50(pretrained=pretrained)

        for param_base, param_target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            param_target.data.copy_(param_base.data)  # initialize
            param_target.requires_grad = False  # not updated by gradient

        self.target_projector = projection_MLP_orchestra(in_dim=self.backbone.output_dim, out_dim=512)

        for param_base, param_target in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_target.data.copy_(param_base.data)  # initialize
            param_target.requires_grad = False  # not updated by gradient

        ### Degeneracy regularization layer (implemented via rotation prediction)
        self.deg_layer = nn.Linear(512, 4)

        ### Centroids [D, N_centroids] and projection memories [D, m_size]
        self.mem_projections = nn.Linear(self.m_size, 512, bias=False)
        self.centroids = nn.Linear(512, self.N_centroids, bias=False) # must be defined second last
        self.local_centroids = nn.Linear(512, self.N_local, bias=False) # must be defined last


    @torch.no_grad()
    def reset_memory(self, data, device='cuda:7'):
        self.train()

        # Save BN parameters to ensure they are not changed when initializing memory
        reset_dict = OrderedDict({k: torch.Tensor(np.array([v.cpu().numpy()])) if (v.shape == ()) else torch.Tensor(v.cpu().numpy()) for k, v in self.state_dict().items() if 'bn' in k})

        # generate feature bank
        proj_bank = []
        n_samples = 0
        for x, _ in data:
            if(n_samples >= self.m_size):
                break
            # Projector representations
            z = F.normalize(self.target_projector(self.target_backbone(x[0].to(device))), dim=1)
            proj_bank.append(z)
            n_samples += x[0].shape[0]

        # Proj_bank: [m_size, D]
        proj_bank = torch.cat(proj_bank, dim=0).contiguous()
        if(n_samples > self.m_size):
            proj_bank = proj_bank[:self.m_size]

        # Save projections: size after saving [D, m_size]
        self.mem_projections.weight.data.copy_(proj_bank.T)

        # Reset BN parameters to original state
        self.load_state_dict(reset_dict, strict=False)


    @torch.no_grad()
    def update_memory(self, F):
        N = F.shape[0]
        # Shift memory [D, m_size]
        self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:, N:].detach().clone()
        # Transpose LHS [D, bsize]
        self.mem_projections.weight.data[:, -N:] = F.T.detach().clone()


    # Target model's update
    @torch.no_grad()
    def update_target(self):
        tau = self.ema_value
        for target, online in zip(self.target_backbone.parameters(), self.backbone.parameters()):
            target.data = (tau) * target.data + (1 - tau) * online.data
        for target, online in zip(self.target_projector.parameters(), self.projector.parameters()):
            target.data = (tau) * target.data + (1 - tau) * online.data


    # Local clustering (happens at the client after every training round; clusters are made equally sized via Sinkhorn-Knopp, satisfying K-anonymity)
    def local_clustering(self, device='cuda:7'):
        # Local centroids: [# of centroids, D]; local clustering input (mem_projections.T): [m_size, D]
        with torch.no_grad():
            Z = self.mem_projections.weight.data.T.detach().clone()
            centroids = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]
            local_iters = 5
            # clustering
            for it in range(local_iters):
                assigns = sknopp(Z @ centroids.T, max_iters=10)
                choice_cluster = torch.argmax(assigns, dim=1)
                for index in range(self.N_local):
                    selected = torch.nonzero(choice_cluster == index).squeeze()
                    selected = torch.index_select(Z, 0, selected)
                    if selected.shape[0] == 0:
                        selected = Z[torch.randint(len(Z), (1,))]
                    centroids[index] = F.normalize(selected.mean(dim=0), dim=0)
                    
        # Save local centroids
        self.local_centroids.weight.data.copy_(centroids.to(device))


    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def global_clustering(self, Z1, nG=1., nL=1.):
        N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]

        # Optimizer setup
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        train_loss = 0.
        total_rounds = 500

        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.centroids(Z1))

            # Zero grad
            optimizer.zero_grad()

            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)

            # Match predicted assignments with SK assignments
            loss = - F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()

            # Train
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1)) # Normalize centroids
                train_loss += loss.item()

            progress_bar(round_idx, total_rounds, 'Loss: %.3f' % (train_loss/(round_idx+1))) 


    # Main training function
    def forward(self, x1, x2, x3=None, deg_labels=None):
        N = x1.shape[0]
        C = self.centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        Z1 = F.normalize(self.projector(self.backbone(x1)), dim=1)
        Z2 = F.normalize(self.projector(self.backbone(x2)), dim=1)

        # Compute online model's assignments 
        cZ2 = Z2 @ C

        # Convert to log-probabilities
        logpZ2 = torch.log(F.softmax(cZ2 / self.T, dim=1))

        # Target outputs [bsize, D]
        with torch.no_grad():
            self.update_target()
            tZ1 = F.normalize(self.target_projector(self.target_backbone(x1)), dim=1)

            # Compute target model's assignments
            cP1 = tZ1 @ C
            tP1 = F.softmax(cP1 / self.T, dim=1)

        # Clustering loss
        L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean()

        # Degeneracy regularization
        deg_preds = self.deg_layer(self.projector(self.backbone(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_cluster + L_deg

        # Update target memory
        with torch.no_grad():
            self.update_memory(tZ1) # New features are [bsize, D]
            
        return L