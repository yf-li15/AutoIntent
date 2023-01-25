import world
from model import HGEncoder
from utils import *
from dataloader import *


class DisenIntentModelManager:
    
    def __init__(self, args, num_labeled_classes, num_unlabeled_classes):
        self.model = HGEncoder(world.config, num_labeled_classes, num_unlabeled_classes)
        self.device = world.device
        self.model.to(self.device)

        self.load_pretrained_model()

        self.freeze_parameters()   # freeze parameters

        self.optimizer = self.get_optimizer(args)

        self.cos = nn.CosineSimilarity(dim=-1)
        
        self.best_eval_score_u = 0
        self.best_scores = None

        # filter  # LP > BP > HP
        self.init_filter(mode='None')
        
        
    def init_filter(self, mode):
        filter_ = np.zeros((1, args.emb_size//2+1))
        if mode == 'LP':
            # low pass (LP)
            for i in range((args.emb_size//2+1)//2):
                filter_[0, i] = 1
        elif mode == 'BP':
            # band pass (BP)
            for i in range((args.emb_size//2+1)//4, args.emb_size//2+1-(args.emb_size//2+1)//4):
                filter_[0, i] = 1
        elif mode == 'HP':
            # high pass (HP)
            for i in range((args.emb_size//2+1)//2, args.emb_size//2+1):
                filter_[0, i] = 1
        else:
            filter_ = np.ones((1, args.emb_size//2+1))
        
        self.filter = torch.tensor(filter_).to(self.device)


    def filter_fft(self, filter, feat):
        feat_f = torch.fft.rfft(feat, dim=-1)
        return  torch.fft.irfft(filter*feat_f, dim=-1)

    def eval(self, args, eval_loader):
        self.model.eval()
        preds=np.array([])
        targets=np.array([])
        for step, (x, label_ids, idx) in enumerate(tqdm(eval_loader, desc="Iteration")):
            uids, lids, tids, cids, negids = x
            uids, lids, tids, cids, negids, label_ids = uids.to(self.device), lids.to(self.device), tids.to(self.device), cids.to(self.device), negids.to(self.device), label_ids.to(self.device)
            # intent_l, intent_t, intent_a, logits_l_l, logits_t_l, logits_a_l, logits_l_u, logits_t_u, logits_a_u, logits_l, logits_u
            _, _, _, _, _, _, _, _, _, logits_l, logits_u  = self.model(uids, lids, tids, cids, negids, label_ids)
            if args.head=='head_l':
                output = logits_l
            else:
                output = logits_u 
            _, pred = output.max(1)
            targets=np.append(targets, label_ids.cpu().numpy())
            preds=np.append(preds, pred.cpu().numpy())

        scores = clustering_score(targets.astype(int), preds.astype(int))
        acc, nmi, ari = scores['ACC'], scores['ARI'], scores['NMI']
        print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
        return acc, scores

    def Pairwise_labeling(self, feat, mask_lb, prob_ulb):
        # add your Pairwise_labeling method here.
        rank_feat = (feat[~mask_lb]).detach()
        feat1, feat2= PairEnum(rank_feat)
        # cosine similarity
        target_ulb = self.cos(feat1, feat2)

        prob1_ulb, prob2_ulb= PairEnum(prob_ulb[~mask_lb])

        return prob1_ulb, prob2_ulb, target_ulb

    def train(self, args, train_loader, labeled_eval_loader, unlabeled_eval_loader, num_labeled_classes):  
 
        wait = 0
        best_model = None
        criterion1 = nn.CrossEntropyLoss() 
        criterion2 = BCE() 
        for epoch in trange(int(args.epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, (x, label_ids, idx) in enumerate(tqdm(train_loader, desc="Iteration")):
                uids, lids, tids, cids, negids = x
                uids, lids, tids, cids, negids, label_ids = uids.to(self.device), lids.to(self.device), tids.to(self.device), cids.to(self.device), negids.to(self.device), label_ids.to(self.device)
                # intent_l, intent_t, intent_a, logits_l_l, logits_t_l, logits_a_l, logits_l_u, logits_t_u, logits_a_u, logits_l, logits_u
                intent_l, intent_t, intent_a, logits_l_l, logits_t_l, logits_a_l, logits_l_u, logits_t_u, logits_a_u, logits_l, logits_u \
                     = self.model(uids, lids, tids, cids, negids, label_ids)


                mask_lb = label_ids<num_labeled_classes
                # disentangled pairwise labeling for unlabeld sampls
                prob_ulb_l, prob_ulb_t, prob_ulb_a = F.softmax(logits_l_u, dim=1), F.softmax(logits_t_u, dim=1), F.softmax(logits_a_u, dim=1)


                loss_ce = criterion1(logits_l[mask_lb], label_ids[mask_lb])
                # location aspect
                intent_l = self.filter_fft(self.filter, intent_l)     # fft
                prob1_ulb_l, prob2_ulb_l, target_ulb_l = self.Pairwise_labeling(intent_l, mask_lb, prob_ulb_l)
                # time aspect
                intent_t = self.filter_fft(self.filter, intent_t)     # fft
                prob1_ulb_t, prob2_ulb_t, target_ulb_t = self.Pairwise_labeling(intent_t, mask_lb, prob_ulb_t)
                # item aspect
                intent_a = self.filter_fft(self.filter, intent_a)     # fft
                prob1_ulb_a, prob2_ulb_a, target_ulb_a = self.Pairwise_labeling(intent_a, mask_lb, prob_ulb_a)
                
        
                loss_bce = criterion2(prob1_ulb_l, prob2_ulb_l, target_ulb_l) + criterion2(prob1_ulb_t, prob2_ulb_t, target_ulb_t) + criterion2(prob1_ulb_a, prob2_ulb_a, target_ulb_a)

                loss = loss_ce + loss_bce

                loss.backward()
                tr_loss += loss.item()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                nb_tr_examples += uids.size(0)
                nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)

            args.head = 'head_l'
            eval_score_l, _ = self.eval(args, labeled_eval_loader)
            print('eval_acc (labeled data):', eval_score_l)

            args.head = 'head_u'
            eval_score_u, scores = self.eval(args, unlabeled_eval_loader)
            print('eval_acc (unlabeled data):', eval_score_u)
            
            if eval_score_u > self.best_eval_score_u:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score_u = eval_score_u
                self.best_scores = scores
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        
        print("Best acc_u:{:.4f}".format(self.best_eval_score_u))  
        print("Best Results:", self.best_scores)
        self.model = best_model
        if args.save_model:
            self.save_model(args)
        

    def get_optimizer(self, args):
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        return optimizer
    
    def save_model(self, args):
        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  
        model_file = os.path.join(args.pretrain_dir, args.dataset+'_ENCODER_WEIGHTS_DisenIntent.pth')
        torch.save(self.save_model.state_dict(), model_file)
    
    def load_pretrained_model(self):
        model_file = os.path.join(args.pretrain_dir, args.dataset+'_ENCODER_WEIGHTS_SupervisedTrain.pth')
        pretrained_dict = torch.load(model_file)
        #classifier_params = ['head_l_l.weight','head_l_l.bias', 'head_t_l.weight','head_t_l.bias', 'head_a_l.weight','head_a_l.bias', \
        #                    'head_l_u.weight','head_l_u.bias', 'head_t_u.weight','head_t_u.bias', 'head_a_u.weight','head_a_u.bias', 'w_intent.weight', 'w_intent.bias']
        #pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
    
    def freeze_parameters(self):
        for name, param in self.model.named_parameters():  
            param.requires_grad = False
            if 'head' in name or 'w_intent' in name:
                param.requires_grad = True



if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    from parse import parse_args
    args = parse_args()
    seed_torch(seed=args.seed)
    # setting
    # intent: 10 labeled + 9 unlabeled
    labeled_list = range(10)
    unlabeled_list = range(10, 19)
    mix_train_loader = LoaderMix(batch_size=args.train_batch_size, mode='train', shuffle=True, labeled_list=labeled_list, unlabeled_list=unlabeled_list)
    labeled_train_loader = Loader(batch_size=args.train_batch_size, mode='train', shuffle=True, target_list = labeled_list)
    unlabeled_eval_loader = Loader(batch_size=args.train_batch_size, mode='eval', shuffle=False, target_list = unlabeled_list)
    unlabeled_eval_loader_test = Loader(batch_size=args.train_batch_size, mode='test', shuffle=False, target_list = unlabeled_list)
    labeled_eval_loader = Loader(batch_size=args.train_batch_size, mode='test', shuffle=False, target_list = labeled_list)
    all_eval_loader = Loader(batch_size=args.train_batch_size, mode='test', shuffle=False, target_list = None)

    # Pretraining Model
    print('DisenIntent-training begin...')
    manager_p = DisenIntentModelManager(args, num_labeled_classes=len(labeled_list), num_unlabeled_classes=len(unlabeled_list))
    manager_p.train(args, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, num_labeled_classes=len(labeled_list))
    print('DisenIntent-training finished!')

