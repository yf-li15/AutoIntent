import world
from model import HGEncoder
from utils import *
from dataloader import *


class PretrainModelManager:
    
    def __init__(self, args):
        self.model = HGEncoder(world.config, mode = 'pretrain')
        self.device = world.device
        self.model.to(self.device)
        
        self.optimizer = self.get_optimizer(args)
        
        self.best_eval_score = 1e12

    def eval(self, args, eval_loader):
        self.model.eval()
        val_loss = 0
        nb_val_examples, nb_val_steps = 0, 0
        for step, (x, label_ids, idx) in enumerate(tqdm(eval_loader, desc="Iteration")):
            uids, lids, tids, cids, negids = x
            uids, lids, tids, cids, negids, label_ids = uids.to(self.device), lids.to(self.device), tids.to(self.device), cids.to(self.device), negids.to(self.device), label_ids.to(self.device)
            with torch.set_grad_enabled(False):
                loss = self.model(uids, lids, tids, cids, negids, label_ids)
                val_loss += loss.item()
                nb_val_examples += uids.size(0)
                nb_val_steps += 1
        
        loss = val_loss / nb_val_steps
        
        return loss


    def train(self, args, train_loader, eval_loader):  
 
        wait = 0
        best_model = None
        for epoch in trange(int(args.epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, (x, label_ids, idx) in enumerate(tqdm(train_loader, desc="Iteration")):
                uids, lids, tids, cids, negids = x
                uids, lids, tids, cids, negids, label_ids = uids.to(self.device), lids.to(self.device), tids.to(self.device), cids.to(self.device), negids.to(self.device), label_ids.to(self.device)
                with torch.set_grad_enabled(True):
                    loss = self.model(uids, lids, tids, cids, negids, label_ids)
                    loss.backward()
                    tr_loss += loss.item()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += uids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.eval(args, eval_loader)
            print('eval_loss', eval_score)
            
            if eval_score < self.best_eval_score:
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
        model_file = os.path.join(args.pretrain_dir, args.dataset+'_ENCODER_WEIGHTS_Pretrain.pth')
        torch.save(self.save_model.state_dict(), model_file)



if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    from parse import parse_args
    args = parse_args()
    seed_torch(seed=args.seed)
    train_loader = Loader(batch_size=args.train_batch_size, mode='train', shuffle=True, target_list = None)
    eval_loader = Loader(batch_size=args.train_batch_size, mode='eval', shuffle=False, target_list = None)

    # Pretraining Model
    print('Pre-training begin...')
    manager_p = PretrainModelManager(args)
    manager_p.train(args, train_loader, eval_loader)
    print('Pre-training finished!')