import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from main.config import TestOptions, setup_model
from run_on_video import clip, txt2clip, vid2clip
from utils.basic_utils import l2_normalize_np_array

parser = argparse.ArgumentParser(description="")
parser.add_argument("--save_dir", type=str, default="./tmp")
parser.add_argument("--resume", type=str, default="./results/model_best_pt_ft.ckpt")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--prompt", type=str, default='interesting')
args = parser.parse_args()

model_version = "ViT-B/32"
output_feat_size = 512
clip_len = 2
overwrite = True
num_decoding_thread = 4
half_precision = False


def load_model():
    opt = TestOptions().parse(args)
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = (
            opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        )
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model


vtg_model = load_model()
clip_model, _ = clip.load(model_version, device=args.gpu_id, jit=False)


def load_data(save_dir):
    vid = np.load(Path(save_dir) / "vid.npz")["features"].astype(np.float32)
    txt = np.load(Path(save_dir) / "txt.npz")["features"].astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    ctx_l = vid.shape[0]

    timestamp = (
        ((torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)
    )

    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef_ed = tef_st + 1.0 / ctx_l
    tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
    vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l


def forward(model, save_dir, query):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir)
    src_vid = src_vid.cuda(args.gpu_id)
    src_txt = src_txt.cuda(args.gpu_id)
    src_vid_mask = src_vid_mask.cuda(args.gpu_id)
    src_txt_mask = src_txt_mask.cuda(args.gpu_id)

    model.eval()
    with torch.no_grad():
        output = model(
            src_vid=src_vid,
            src_txt=src_txt,
            src_vid_mask=src_vid_mask,
            src_txt_mask=src_txt_mask,
        )

    # prepare the model prediction
    # pred_logits = output["pred_logits"][0].cpu()
    # pred_spans = output["pred_spans"][0].cpu()
    pred_saliency = output["saliency_scores"].cpu()

    pd.DataFrame({'saliency': pred_saliency.flatten()}).to_csv(save_dir + '/sailency.csv', index=False)


def extract_vid(vid_path):
    vid_features = vid2clip(clip_model, vid_path, args.save_dir)
    return vid_features


def extract_txt(txt):
    txt_features = txt2clip(clip_model, txt, args.save_dir)
    return txt_features


def main(path):
    if False:
        extract_txt('interesting')
        extract_vid(path)
    forward(vtg_model, args.save_dir, args.prompt)

if __name__ == '__main__':
    main('/home/ubuntu/UniVTG/data/1260064971/1260064971.mp4')
