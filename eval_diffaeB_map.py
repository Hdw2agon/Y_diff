import os
import importlib
from utils.args import make_args


TESTER_MAP = {
    'backbone': 'model.tester_backbone',
    'y_diff': 'model.tester_Y_diff',
}


def get_tester_class(version_alias):
    if version_alias not in TESTER_MAP:
        raise ValueError(f"Unknown tester version: {version_alias}. Available versions are: {list(TESTER_MAP.keys())}")

    module_path = TESTER_MAP[version_alias]
    try:
        tester_module = importlib.import_module(module_path)
        return tester_module.DiffFSTester
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import DiffFSTester from {module_path}: {e}")


def main(args):
    DiffFSTester = get_tester_class(args.tester_version)
    print(f"Using Tester version: {args.tester_version} from {TESTER_MAP[args.tester_version]}")

    tester = DiffFSTester(args)
    tester.infer_image_all_multiGPU()


if __name__ == '__main__':
    args = make_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")

    main(args)