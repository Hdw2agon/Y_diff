import os
import importlib
from utils.args import make_args

TRAINER_MAP = {
    'backbone': 'model.trainer_teacher',
    'fm_student': 'model.trainer_fm_student',
}

def get_trainer_class(version_alias):
    if version_alias not in TRAINER_MAP:
        raise ValueError(f"Unknown trainer version: {version_alias}. Available versions are: {list(TRAINER_MAP.keys())}")
    
    module_path = TRAINER_MAP[version_alias]
    try:
        trainer_module = importlib.import_module(module_path) 
        return trainer_module.DiffFSTrainer 
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import DiffFSTrainer from {module_path}: {e}")


def main(args):
    DiffFSTrainer = get_trainer_class(args.trainer_version)
    print(f"Using Trainer version: {args.trainer_version} from {TRAINER_MAP[args.trainer_version]}")
    
    trainer = DiffFSTrainer(args)
    trainer.train()


if __name__ == '__main__':
    args = make_args()
    
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")
        
    main(args)