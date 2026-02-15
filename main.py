import argparse
import fit_data, train_model, eval_model

def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)
    parser.add_argument('--mode', default='fit', choices=['fit', 'train', 'eval'], type=str)
    
    # Fit params
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int) # 1000 for eval
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    
    # Training params
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--batch_size", default=32, type=int) #32
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_freq", default=250, type=int)
    parser.add_argument("--load_checkpoint", action="store_true") 
    parser.add_argument('--load_feat', action='store_true') 
    
    # Evaluation params
    parser.add_argument('--vis_freq', default=50, type=int)
    parser.add_argument('--interpret', action='store_true', help="Enable interpretation mode")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if args.mode == "fit":
        fit_data.train_model(args)
    elif args.mode == "train":
        train_model.train_model(args)
    elif args.mode == "eval": #Run w/t --interpret for better visualization 
        eval_model.evaluate_model(args)
    else:
        raise ValueError(f"Unsupported Mode")

    