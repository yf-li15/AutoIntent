
import world
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
    def attnw(self):
        return self.attnw
    
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        self.attnw = beta
        return (beta * z).sum(1)



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getRating(self, users, locations, times):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, locations, times, pos, neg):
        """
        Parameters:
            users: users list 
            locations: locations list
            times: times list
            pos: positive items for corresponding Scenes
            neg: negative items for corresponding Scenes
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class HGEncoder(BasicModel):
    def __init__(self, 
                 config:dict,
                 num_labeled_classes=None, 
                 num_unlabeled_classes=None,
                 mode=None
                 ):
        """
        Hypergraph Encoder (HGEncoder) for intent feature generation.
        main task: intent classification;
        aux  task: user comsumption prediction (BPR loss).
        config: parameters
        """
        super(HGEncoder, self).__init__()
        self.config = config
        self.num_users = self.config['num_users']
        self.num_locations = self.config['num_locations']
        self.num_times = self.config['num_times']
        self.num_activities  = self.config['num_activities']
        self.latent_dim = self.config['emb_size']
        self.n_layers = self.config['layer']
        self.use_drop = self.config['use_drop']
        self.keep_prob = 1 - self.config['droprate']
        self.dropout = self.config['droprate']
        self.mode = mode
        self.device = world.device
        if not self.mode == 'pretrain':
            self.num_labeled_classes = num_labeled_classes          # labeled intent classes number
            self.num_unlabeled_classes = num_unlabeled_classes      # unlabeled intent classes number
        
        # Load dual hypergraph matrix for U-L, U-T, U-A
        HG_ul = sp.load_npz(world.DATA_PATH + '/HG_ul.npz')
        self.HG_ul = self._convert_sp_mat_to_sp_tensor(HG_ul)
        self.HG_ul = self.HG_ul.coalesce().to(world.device) 
        HG_ut = sp.load_npz(world.DATA_PATH + '/HG_ut.npz')
        self.HG_ut = self._convert_sp_mat_to_sp_tensor(HG_ut)
        self.HG_ut = self.HG_ut.coalesce().to(world.device) 
        HG_ua = sp.load_npz(world.DATA_PATH + '/HG_ua.npz')
        self.HG_ua = self._convert_sp_mat_to_sp_tensor(HG_ua)
        self.HG_ua = self.HG_ua.coalesce().to(world.device) 

        HG_l = sp.load_npz(world.DATA_PATH + '/HG_l.npz')
        self.HG_l = self._convert_sp_mat_to_sp_tensor(HG_l)
        self.HG_l = self.HG_l.coalesce().to(world.device) 
        HG_t = sp.load_npz(world.DATA_PATH + '/HG_t.npz')
        self.HG_t = self._convert_sp_mat_to_sp_tensor(HG_t)
        self.HG_t = self.HG_t.coalesce().to(world.device) 
        HG_a = sp.load_npz(world.DATA_PATH + '/HG_a.npz')
        self.HG_a = self._convert_sp_mat_to_sp_tensor(HG_a)
        self.HG_a = self.HG_a.coalesce().to(world.device) 

        # normlized transition hypergraph
        norm_VtoE_lu = sp.load_npz(world.DATA_PATH + '/norm_vtoe_lu.npz')
        self.VtoE_lu = self._convert_sp_mat_to_sp_tensor(norm_VtoE_lu)
        self.VtoE_lu = self.VtoE_lu.coalesce().to(world.device) 
        norm_VtoE_tu = sp.load_npz(world.DATA_PATH + '/norm_vtoe_tu.npz')
        self.VtoE_tu = self._convert_sp_mat_to_sp_tensor(norm_VtoE_tu)
        self.VtoE_tu = self.VtoE_tu.coalesce().to(world.device) 
        norm_VtoE_au = sp.load_npz(world.DATA_PATH + '/norm_vtoe_au.npz')
        self.VtoE_au = self._convert_sp_mat_to_sp_tensor(norm_VtoE_au)
        self.VtoE_au = self.VtoE_au.coalesce().to(world.device) 

        norm_VtoE_ul = sp.load_npz(world.DATA_PATH + '/norm_vtoe_ul.npz')
        self.VtoE_ul = self._convert_sp_mat_to_sp_tensor(norm_VtoE_ul)
        self.VtoE_ul = self.VtoE_ul.coalesce().to(world.device) 
        norm_VtoE_ut = sp.load_npz(world.DATA_PATH + '/norm_vtoe_ut.npz')
        self.VtoE_ut = self._convert_sp_mat_to_sp_tensor(norm_VtoE_ut)
        self.VtoE_ut = self.VtoE_ut.coalesce().to(world.device) 
        norm_VtoE_ua = sp.load_npz(world.DATA_PATH + '/norm_vtoe_ua.npz')
        self.VtoE_ua = self._convert_sp_mat_to_sp_tensor(norm_VtoE_ua)
        self.VtoE_ua = self.VtoE_ua.coalesce().to(world.device)

        self.__init__weight()

    def __init__weight(self):
        
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_locations, embedding_dim=self.latent_dim)
        self.embedding_time = torch.nn.Embedding(
            num_embeddings=self.num_times, embedding_dim=self.latent_dim)
        self.embedding_activity  = torch.nn.Embedding(
            num_embeddings=self.num_activities,  embedding_dim=self.latent_dim)
        
        """
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_location.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_time.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_activity.weight,  gain=1)
        print('use xavier initilizer')
        """
        # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_location.weight, std=0.1)
        nn.init.normal_(self.embedding_time.weight, std=0.1)
        nn.init.normal_(self.embedding_activity.weight,  std=0.1)
        world.cprint('use NORMAL distribution initilizer')
            
        
        self.f = nn.Sigmoid() # for final prediction
        self.disen_l = nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        self.disen_t = nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        self.disen_a = nn.Linear(self.latent_dim, self.latent_dim, bias=True)

        
        if not self.mode == 'pretrain':
            # MLP for intent features
            self.MLP_l = nn.Linear(self.latent_dim*2, self.latent_dim, bias=True)
            self.MLP_t = nn.Linear(self.latent_dim*2, self.latent_dim, bias=True)
            self.MLP_a = nn.Linear(self.latent_dim*2, self.latent_dim, bias=True)
            # head for labeled samples
            self.head_l_l = nn.Linear(self.latent_dim, self.num_labeled_classes)
            self.head_t_l = nn.Linear(self.latent_dim, self.num_labeled_classes)
            self.head_a_l = nn.Linear(self.latent_dim, self.num_labeled_classes)
            # head for unlabeled samples
            self.head_l_u = nn.Linear(self.latent_dim, self.num_labeled_classes)
            self.head_t_u = nn.Linear(self.latent_dim, self.num_labeled_classes)
            self.head_a_u = nn.Linear(self.latent_dim, self.num_labeled_classes)
            # map feature for final predict
            self.w_intent = nn.Linear(self.latent_dim, 1)



        print(f"HGEncoder is already to go!")
   
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def computer(self):
        """
        Calculate Joint-HGConv on Hypergraph: Combine Intra- and Inter-graph message passing in dual hypergraphs
        """      
        users_emb = self.embedding_user.weight
        locations_emb = self.embedding_location.weight
        times_emb = self.embedding_time.weight
        activities_emb  = self.embedding_activity.weight
        
        # get hypergraph 
        HG_ul, HG_ut, HG_ua, HG_l, HG_t, HG_a, VtoE_lu, VtoE_tu, VtoE_au, VtoE_ul, VtoE_ut, VtoE_ua = \
        self.HG_ul, self.HG_ut, self.HG_ua, self.HG_l, self.HG_t, self.HG_a, self.VtoE_lu, self.VtoE_tu, self.VtoE_au, self.VtoE_ul, self.VtoE_ut, self.VtoE_ua

        # user embedding transformation (location-, time-, and item-aware preferences)
        u_emb_l = self.disen_l(users_emb)
        u_emb_t = self.disen_t(users_emb)
        u_emb_a = self.disen_a(users_emb)
        
          
        u_emb_l_all = [u_emb_l]
        u_emb_t_all = [u_emb_t]
        u_emb_a_all = [u_emb_a]

        l_emb = locations_emb
        l_emb_all = [l_emb]
        t_emb = times_emb
        t_emb_all = [t_emb]
        a_emb = activities_emb
        a_emb_all = [a_emb]

      
        for k in range(self.n_layers):
            # Joint-HGConv
            u_emb_l1 = torch.sparse.mm(HG_ul, u_emb_l) + torch.sparse.mm(VtoE_lu, l_emb)
            l_emb1 = torch.sparse.mm(HG_l, l_emb) + torch.sparse.mm(VtoE_ul, u_emb_l)

            u_emb_t1 = torch.sparse.mm(HG_ut, u_emb_t) + torch.sparse.mm(VtoE_tu, t_emb)
            t_emb1 = torch.sparse.mm(HG_t, t_emb) + torch.sparse.mm(VtoE_ut, u_emb_t)

            u_emb_a1 = torch.sparse.mm(HG_ua, u_emb_a) + torch.sparse.mm(VtoE_au, a_emb)
            a_emb1 = torch.sparse.mm(HG_a, a_emb) + torch.sparse.mm(VtoE_ua, u_emb_a)

            # update embeddings
            u_emb_l = u_emb_l1
            l_emb = l_emb1
            u_emb_t = u_emb_t1
            t_emb = t_emb1
            u_emb_a = u_emb_a1
            a_emb = a_emb1

            u_emb_l_all.append(u_emb_l)
            u_emb_t_all.append(u_emb_t)
            u_emb_a_all.append(u_emb_a)
            l_emb_all.append(l_emb)
            t_emb_all.append(t_emb)
            a_emb_all.append(a_emb)

        # read-out
        users_l = torch.mean(torch.stack(u_emb_l_all, dim=1), dim=1)
        users_t = torch.mean(torch.stack(u_emb_t_all, dim=1), dim=1)
        users_a = torch.mean(torch.stack(u_emb_a_all, dim=1), dim=1)
        
        locations = torch.mean(torch.stack(l_emb_all, dim=1), dim=1)
        times = torch.mean(torch.stack(t_emb_all, dim=1), dim=1)
        activities = torch.mean(torch.stack(a_emb_all, dim=1), dim=1)
        
        return users_l, users_t, users_a, locations, times, activities 

    
    def getRating(self, users, locations, times):
        all_users_l, all_users_t, all_users_a, all_locations, all_times, all_activities = self.computer()
        users = users.long()
        locations = locations.long()
        times = times.long()
        users_emb_l = all_users_l[users]
        users_emb_t = all_users_t[users]
        users_emb_a = all_users_a[users]
        locations_emb = all_locations[locations]
        times_emb = all_times[times]
        activities_emb  = all_activities
        scores = torch.sum(users_emb_l*locations_emb + users_emb_t*times_emb, dim=1, keepdim=True) + torch.matmul(users_emb_a, activities_emb.t())
        #scores = torch.matmul(users_emb_l*locations_emb*users_emb_t*times_emb*users_emb_a, activities_emb.t())
        return self.f(scores)

    def getEmbedding(self, users, locations, times, pos, neg):
        all_users_l, all_users_t, all_users_a, all_locations, all_times, all_activities = self.computer()
        users_emb_l = all_users_l[users]
        users_emb_t = all_users_t[users]
        users_emb_a = all_users_a[users]
        locations_emb = all_locations[locations]
        times_emb = all_times[times]
        pos_emb = all_activities[pos]
        neg_emb = all_activities[neg]
        
        users_emb_ego = self.embedding_user(users)
        locations_emb_ego = self.embedding_location(locations)
        times_emb_ego = self.embedding_time(times)
        pos_emb_ego   = self.embedding_activity(pos)
        neg_emb_ego   = self.embedding_activity(neg)

        return users_emb_l, users_emb_t, users_emb_a, locations_emb, times_emb, pos_emb, neg_emb, users_emb_ego, locations_emb_ego, times_emb_ego, pos_emb_ego, neg_emb_ego
    
    
    def bpr_loss(self, users_emb_l, users_emb_t, users_emb_a, locations_emb, times_emb, pos_emb, neg_emb, userEmb0, baseEmb0, timeEmb0, posEmb0, negEmb0):
        # BPR loss to optimize aux task
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         baseEmb0.norm(2).pow(2) + 
                         timeEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users_emb_l))

        pos_scores= torch.sum(users_emb_l*locations_emb + users_emb_t*times_emb + users_emb_a*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb_l*locations_emb + users_emb_t*times_emb + users_emb_a*neg_emb, dim=1) #[2048]
        
        BPR_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return BPR_loss, reg_loss
    

    def forward(self, users, locations, times, pos, neg, labels=None):
        """
        Input:(users, locations, times, pos, neg, labels, mode, feature_ext)
        """
        # compute embedding
        (users_emb_l, users_emb_t, users_emb_a, locations_emb, times_emb, pos_emb, neg_emb, 
        userEmb0, baseEmb0, timeEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), locations.long(),times.long(), pos.long(), neg.long())

        
        if self.mode == 'pretrain':
            bpr_loss, reg_loss = self.bpr_loss(users_emb_l, users_emb_t, users_emb_a, locations_emb, times_emb, pos_emb, neg_emb, userEmb0, baseEmb0, timeEmb0, posEmb0, negEmb0)
            total_loss = bpr_loss + self.config['decay'] * reg_loss
            return total_loss
        
        else:
            # intents feature in three aspects
            intent_l = self.MLP_l(torch.cat([users_emb_l, locations_emb], dim=-1))
            intent_t = self.MLP_t(torch.cat([users_emb_t, times_emb], dim=-1))
            intent_a = self.MLP_a(torch.cat([users_emb_a, pos_emb], dim=-1))
            
            # calculate logits
            logits_l_l = self.head_l_l(intent_l)
            logits_t_l = self.head_t_l(intent_t)
            logits_a_l = self.head_a_l(intent_a)
            logits_l_u = self.head_l_u(intent_l)
            logits_t_u = self.head_t_u(intent_t)
            logits_a_u = self.head_a_u(intent_a)



            intent_map = self.w_intent(torch.stack([intent_l, intent_t, intent_a], dim=1))  # B, 3, 1
            beta = F.softmax(intent_map, dim=1)

            logits_l = torch.sum(torch.stack([logits_l_l, logits_t_l, logits_a_l], dim=1) * beta, dim=1, keepdim=False)  # B, class_num
            logits_u = torch.sum(torch.stack([logits_l_u, logits_t_u, logits_a_u], dim=1) * beta, dim=1, keepdim=False)  # B, class_num

            
            return intent_l, intent_t, intent_a, logits_l_l, logits_t_l, logits_a_l, logits_l_u, logits_t_u, logits_a_u, logits_l, logits_u


if __name__ == '__main__':
    model = HGEncoder(world.config, num_labeled_classes=10, num_unlabeled_classes=9)
    print(model)
    for name, param in model.named_parameters():
        print(name)