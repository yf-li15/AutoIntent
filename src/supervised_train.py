import world
from model import HGEncoder
from utils import *
from dataloader import *


class SupervisedModelManager:
    
    def __init__(self, args, num_labeled_classes, num_unlabeled_classes):
        self.model = HGEncoder(world.config, num_labeled_classes, num_unlabeled_classes)
        self.device = world.device
        self.model.to(self.device)

        self.load_pretrained_model()

        self.freeze_parameters()   # freeze parameters
        
        self.optimizer = self.get_optimizer(args)
        
        self.best_eval_score = 0

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
        return acc 


    def train(self, args, train_loader, eval_loader):  
 
        wait = 0
        best_model = None
        criterion1 = nn.CrossEntropyLoss() 
        for epoch in trange(int(args.epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, (x, label_ids, idx) in enumerate(tqdm(train_loader, desc="Iteration")):
                uids, lids, tids, cids, negids = x
                uids, lids, tids, cids, negids, label_ids = uids.to(self.device), lids.to(self.device), tids.to(self.device), cids.to(self.device), negids.to(self.device), label_ids.to(self.device)
                # intent_l, intent_t, intent_a, logits_l_l, logits_t_l, logits_a_l, logits_l_u, logits_t_u, logits_a_u, logits_l, logits_u
                _, _, _, _, _, _, _, _, _, logits_l,  _  = self.model(uids, lids, tids, cids, negids, label_ids)
                loss = criterion1(logits_l, label_ids)
                loss.backward()
                tr_loss += loss.item()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                nb_tr_examples += uids.size(0)
                nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)

            args.head = 'head_l'
            
            eval_score = self.eval(args, eval_loader)
            print('eval_acc', eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
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
        model_file = os.path.join(args.pretrain_dir, args.dataset+'_ENCODER_WEIGHTS_SupervisedTrain.pth')
        torch.save(self.save_model.state_dict(), model_file)
    
    def load_pretrained_model(self):
        model_file = os.path.join(args.pretrain_dir, args.dataset+'_ENCODER_WEIGHTS_Pretrain.pth')
        pretrained_dict = torch.load(model_file)
        classifier_params = ['head_l_l.weight','head_l_l.bias', 'head_t_l.weight','head_t_l.bias', 'head_a_l.weight','head_a_l.bias', \
                            'head_l_u.weight','head_l_u.bias', 'head_t_u.weight','head_t_u.bias', 'head_a_u.weight','head_a_u.bias', 'w_intent.weight', 'w_intent.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
    
    def freeze_parameters(self):
        for name, param in self.model.named_parameters():  
            param.requires_grad = False
            if 'head' in name or 'w_intent' in name:
                print(name)
                param.requires_grad = True



if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    from parse import parse_args
    args = parse_args()
    seed_torch(seed=args.seed)
    # setting
    labeled_list = range(10)
    unlabeled_list = range(10, 19)

    labeled_train_loader = Loader(batch_size=args.train_batch_size, mode='train', shuffle=True, target_list = labeled_list)
    labeled_eval_loader = Loader(batch_size=args.train_batch_size, mode='eval', shuffle=False, target_list = labeled_list)

    # Pretraining Model
    print('Supervised-training begin...')
    manager_p = SupervisedModelManager(args, num_labeled_classes=len(labeled_list), num_unlabeled_classes=len(unlabeled_list))
    manager_p.train(args, labeled_train_loader, labeled_eval_loader)
    print('Supervised-training finished!')