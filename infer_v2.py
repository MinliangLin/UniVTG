import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from main.config import TestOptions, setup_model
from run_on_video import clip, txt2clip, vid2clip
from utils.basic_utils import l2_normalize_np_array

parser = argparse.ArgumentParser(description="")
parser.add_argument("--save_dir", type=Path, default="./tmp")
parser.add_argument("--resume", type=str, default="./results/model_best_pt_ft.ckpt")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--clip_len", type=float, default=1)
parser.add_argument("--query", type=str, default="flower")
args = parser.parse_args()


def load_vtg_model():
    opt = TestOptions().parse(args)
    cudnn.benchmark = True
    cudnn.deterministic = False
    opt.lr_warmup = -1
    model, _, _, _ = setup_model(opt)
    return model


vtg_model = load_vtg_model()
clip_model, _ = clip.load("ViT-B/32", device=args.gpu_id, jit=False)


def load_data(save_dir):
    vid = np.load(save_dir / "vid.npz")["features"].astype(np.float32)
    txt = np.load(save_dir / "txt.npz")["features"].astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    ctx_l = vid.shape[0]

    timestamp = (
        ((torch.arange(0, ctx_l) + args.clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)
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


def infer(model, out_path):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(out_path.parent)
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

    pred_saliency = output["saliency_scores"].cpu()
    pred_logits = output["pred_logits"][0].cpu()
    print(output["pred_logits"].shape)
    df = pd.DataFrame(
        {"saliency": pred_saliency.flatten(), "logit": pred_logits.flatten()}
    )
    df.to_csv(out_path, index=False)


def extract_vid(vid_path, save_dir):
    if not (save_dir / "vid.npz").exists():
        vid_features = vid2clip(
            clip_model, str(vid_path), str(save_dir), clip_len=args.clip_len
        )
        return vid_features


def extract_txt(txt, save_dir):
    if not (save_dir / "txt.npz").exists():
        txt_features = txt2clip(clip_model, txt, str(save_dir))
        return txt_features


def main():
    df = pd.read_csv('./run.csv')
    for row in df.itertuples():
        content_id = Path(row.path).stem
        save_dir = args.save_dir / content_id
        save_dir.mkdir(exist_ok=True)
        out_csv = save_dir / "saliency_v3.csv"
        if out_csv.exists():
            print(out_csv, 'exists')
            # continue
        print(out_csv)
        extract_txt(args.query, args.save_dir)
        extract_vid('data/data2022/' + row.path, save_dir)
        infer(vtg_model, out_csv)


if __name__ == "__main__":
    main()
