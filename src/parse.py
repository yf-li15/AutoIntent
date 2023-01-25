
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Intent Discovery in Meituan")
    parser.add_argument('--dataset', type=str,default="Beijing",
                        help="[Beijing, Shanghai]")
    parser.add_argument("--save_results_path", type=str, default='results', help="The path to save results.")
    parser.add_argument('--train_batch_size', type=int,default=2048,
                        help="the batch size for training procedure")
    parser.add_argument('--emb_size', type=int,default=64,
                        help="the embedding size of the model")
    parser.add_argument('--layer', type=int,default=1,
                        help="the layer num of Hypergraph")
    parser.add_argument('--lr', type=float,default=1e-3,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--gamma', type=float,default=3e-3,
                        help="the weight for Independent constraint")               
    parser.add_argument('--use_drop', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--droprate', type=float,default=0.4,
                        help="dropout rate")
    parser.add_argument('--test_batch_size', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument("--pretrain_dir", default='pretrain_models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 
    parser.add_argument("--save_model", action="store_true", help="Save trained model.")
    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes.")
    parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The factor (magnification) of the number of clusters K.")
    parser.add_argument("--labeled_ratio", default=0.5, type=float, help="The ratio of labeled samples in the training set.")
    parser.add_argument('--gpu_id', type=str,default="0")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=50)
    parser.add_argument('--multicore', type=int, default=1, help='whether use multiprocessing or not in test')
    parser.add_argument("--pretrain", action="store_true", help="Pre-train the model with labeled data.")
    parser.add_argument("--wait_patient", default=5, type=int,
                        help="Patient steps for Early Stop.") 
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--topk', default=3, type=int)
    parser.add_argument('--IL', action='store_true', default=False, help='w/ incremental learning')
    parser.add_argument('--model', type=str, default='DisenIntent', help='model name')
    parser.add_argument('--use_trans', action='store_true', default=False, help='use trans in GCN')
    parser.add_argument('--use_acf', action='store_true', default=False, help='use activate function in GCN')
    parser.add_argument('--earlystop', action='store_true', default=False, help='use earlystop')
    return parser.parse_args()