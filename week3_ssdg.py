import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from metrics import compute_metrics, save_plots_and_metrics
from model_vfm import build_dinov2_fas
from vfm_data import (
    MICO_DATASETS,
    VFMImageDataset,
    aggregate_video_scores,
    build_domain_class_balanced_sampler,
    evenly_subsample_frames_per_video,
    scan_vfm_domain,
    split_train_val_by_group,
)


class _GRL(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    return _GRL.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, dim, n_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_domains),
        )

    def forward(self, x, grl_lambda=1.0):
        return self.net(grad_reverse(x, grl_lambda))


def batch_hard_group_triplet(features, group_labels, margin=0.1):
    x = F.normalize(features.float(), dim=1)
    dist = torch.cdist(x, x, p=2)
    labels = group_labels.view(-1)
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same = same & ~eye
    diff = ~labels[:, None].eq(labels[None, :])

    losses = []
    for i in range(len(labels)):
        pos = dist[i][same[i]]
        neg = dist[i][diff[i]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        hardest_pos = pos.max()
        hardest_neg = neg.min()
        losses.append(F.relu(hardest_pos - hardest_neg + margin))
    if not losses:
        return features.sum() * 0.0
    return torch.stack(losses).mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--target_dataset", required=True, choices=MICO_DATASETS)
    p.add_argument("--output_dir", default="outputs_week3_ssdg")
    p.add_argument("--eval_dir", default="outputs_week3_ssdg_eval")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--freeze_first_blocks", type=int, default=11)
    p.add_argument("--backbone_lr", type=float, default=3e-6)
    p.add_argument("--head_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--triplet_weight", type=float, default=1.0)
    p.add_argument("--triplet_margin", type=float, default=0.1)
    p.add_argument("--ad_weight", type=float, default=0.2)
    p.add_argument("--grl_lambda", type=float, default=1.0)
    p.add_argument("--max_frames_per_video", type=int, default=20)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_model(model, loader, device, amp):
    model.eval()
    ys, scores, vids, domains = [], [], [], []
    with torch.no_grad():
        for x, y, _, _, d, v in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            ys.extend(y.tolist()); scores.extend(prob.cpu().tolist())
            vids.extend(list(v)); domains.extend(list(d))

    _, vy, vs = aggregate_video_scores(ys, scores, vids)
    overall = compute_metrics(vy, vs)
    by_domain = {}
    for d in sorted(set(domains)):
        idx = [i for i, x in enumerate(domains) if x == d]
        dy = [ys[i] for i in idx]; ds = [scores[i] for i in idx]; dv = [vids[i] for i in idx]
        _, dvy, dvs = aggregate_video_scores(dy, ds, dv)
        by_domain[d] = compute_metrics(dvy, dvs)
    aucs = [m["auc"] for m in by_domain.values()]
    return overall, by_domain, min(aucs), float(np.mean(aucs))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    sources = [d for d in MICO_DATASETS if d != args.target_dataset]
    domain_to_id = {d:i for i,d in enumerate(sources)}

    train_s, val_s = [], []
    for i, d in enumerate(sources):
        all_s = scan_vfm_domain(args.data_root, d)
        tr, va = split_train_val_by_group(all_s, val_ratio=0.10, seed=args.seed + 17*i)
        train_s += evenly_subsample_frames_per_video(tr, args.max_frames_per_video)
        val_s += evenly_subsample_frames_per_video(va, args.max_frames_per_video)

    sampler = build_domain_class_balanced_sampler(train_s)
    train_ds = VFMImageDataset(train_s, train=True, image_size=args.image_size)
    val_ds = VFMImageDataset(val_s, train=False, image_size=args.image_size)
    kw = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers>0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=max(32,args.batch_size), shuffle=False, **kw)

    model = build_dinov2_fas(
        model_name="dinov2_vitb14_reg",
        pretrained=True,
        feature_mode="cls",
        freeze_first_blocks=args.freeze_first_blocks,
    ).to(device)
    dim = model.classifier[-1].in_features
    discriminator = DomainDiscriminator(dim, len(sources)).to(device)

    params = [
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")], "lr":args.backbone_lr},
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")], "lr":args.head_lr},
        {"params":discriminator.parameters(), "lr":args.head_lr},
    ]
    optimizer = AdamW(params, weight_decay=args.weight_decay)
    total_updates = math.ceil(len(train_loader)/args.grad_accum) * args.epochs
    scheduler = LambdaLR(optimizer, lambda s: 0.5*(1+math.cos(math.pi*min(s/max(total_updates,1),1.0))))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    out = Path(args.output_dir)/args.target_dataset
    out.mkdir(parents=True, exist_ok=True)
    history=[]; best_worst=-1.; best_mean=-1.; bad=0

    for epoch in range(1,args.epochs+1):
        model.train(); discriminator.train(); optimizer.zero_grad(set_to_none=True)
        sums=Counter(); n=0
        for step,(x,y,_,_,domains,_) in enumerate(train_loader,1):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            dom=torch.tensor([domain_to_id[d] for d in domains],device=device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                feats=model.extract_features(x)
                logits=model.classifier(feats)
                cls=F.cross_entropy(logits,y)
                pseudo=torch.where(y==1,torch.zeros_like(y),dom+1)
                tri=batch_hard_group_triplet(feats,pseudo,args.triplet_margin)
                live=y==1
                if live.sum()>1:
                    dom_logits=discriminator(feats[live],args.grl_lambda)
                    adv=F.cross_entropy(dom_logits,dom[live])
                else:
                    adv=feats.sum()*0.0
                raw=cls+args.triplet_weight*tri+args.ad_weight*adv
                loss=raw/args.grad_accum
            scaler.scale(loss).backward()
            if step%args.grad_accum==0 or step==len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(discriminator.parameters()),1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            bs=x.size(0); n+=bs
            sums["loss"]+=raw.item()*bs; sums["cls"]+=cls.item()*bs; sums["tri"]+=tri.item()*bs; sums["adv"]+=adv.item()*bs

        overall,by_domain,worst,mean=eval_model(model,val_loader,device,amp)
        row={"epoch":epoch,"loss":sums["loss"]/n,"cls_loss":sums["cls"]/n,"triplet_loss":sums["tri"]/n,"ad_loss":sums["adv"]/n,
             "val_video_auc":overall["auc"],"val_video_eer":overall["eer"],"val_worst_domain_auc":worst,"val_mean_domain_auc":mean,
             "val_by_domain":{d:{"auc":m["auc"],"eer":m["eer"]} for d,m in by_domain.items()}}
        history.append(row); print(json.dumps(row,ensure_ascii=False))
        improved=worst>best_worst+1e-6 or (abs(worst-best_worst)<=1e-6 and mean>best_mean+1e-6)
        if improved:
            best_worst,best_mean=worst,mean; bad=0
            torch.save({"model":model.state_dict(),"args":vars(args),"source_datasets":sources,"target_dataset":args.target_dataset},out/"best.pth")
            print(f"[BEST] epoch={epoch} worst_domain_auc={worst:.6f} mean_domain_auc={mean:.6f}")
        else:
            bad+=1
            if bad>=args.patience:
                print(f"Early stopping at epoch {epoch}."); break

    (out/"history.json").write_text(json.dumps(history,indent=2,ensure_ascii=False),encoding="utf-8")

    ckpt=torch.load(out/"best.pth",map_location="cpu"); model.load_state_dict(ckpt["model"]); model.eval()
    target=scan_vfm_domain(args.data_root,args.target_dataset)
    tds=VFMImageDataset(target,train=False,image_size=args.image_size)
    tl=DataLoader(tds,batch_size=64,shuffle=False,**kw)
    ys=[];sc=[];vids=[]
    with torch.no_grad():
        for x,y,_,_,_,v in tl:
            x=x.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                p=torch.softmax(model(x).float(),dim=1)[:,1]
                pf=torch.softmax(model(torch.flip(x,dims=[3])).float(),dim=1)[:,1]
                p=(p+pf)/2
            ys+=y.tolist(); sc+=p.cpu().tolist(); vids+=list(v)
    eval_out=Path(args.eval_dir)/args.target_dataset
    fm=save_plots_and_metrics(ys,sc,eval_out/"frame_level",prefix="frame")
    _,vy,vs=aggregate_video_scores(ys,sc,vids)
    vm=save_plots_and_metrics(vy,vs,eval_out/"video_level",prefix="video")
    summary={"method":"SSDG-style on DINOv2-Reg","sources":sources,"target":args.target_dataset,"frame_metrics":fm,"video_metrics":vm}
    (eval_out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"FINAL Frame AUC={fm['auc']:.6f} VIDEO AUC={vm['auc']:.6f} VIDEO EER={vm['eer']:.6f}")


if __name__=="__main__":
    main()
