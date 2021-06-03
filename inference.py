import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import TestDataset, MaskBaseDataset


def load_model(module_name, saved_model, device, file_name):
    model_cls = getattr(import_module("model"), module_name)
    model = model_cls(
        model_name=args.model_name
    )
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, file_name)
    model.load_state_dict(torch.load(model_path, map_location=device))

#     print(model)
    return model

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    print("예측을 시작합니다.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    oof_pred = None
    n_splits = len(os.listdir("./model/Kfold/")[1:])
    for k_model in os.listdir("./model/Kfold/")[1:]:
        model = load_model("MyModel",model_dir, device, k_model).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            print("모델에 값을 넣고 있습니다.")
            for idx, images in enumerate(loader):
                images = images.to(device)

                pred = model(images)
                preds.extend(pred.cpu().numpy())
            fold_pred = np.array(preds)
            if oof_pred is None:
                oof_pred = fold_pred / n_splits
            else:
                oof_pred += fold_pred / n_splits
    
    info['ans'] = np.argmax(oof_pred, axis=1)
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
#     parser.add_argument('--model', type=str, default='MyModel', help='model type (default: BaseModel)')
    parser.add_argument('--model_name', type=str, default='vgg19', help='model name (default: vgg11)')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
