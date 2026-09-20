import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from datasets import scan_dataset, strip_frame_suffix
from metrics import compute_metrics, save_plots_and_metrics
from model_vfm import build_dinov2_fas
from vfm_data import MICO_DATASETS, aggregate_video_scores, split_train_val_by_group


MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)


def video_id(path,dataset):
    return f"{dataset}:{strip_frame_suffix(Path(path))}"


def build_sequences(data_root,dataset,max_sequences=10):
    raw=scan_dataset(Path(data_root)/dataset)
    vids=defaultdict(list)
    for p,y,g in raw:
        vids[video_id(p,dataset)].append((p,y,g,dataset,video_id(p,dataset)))
    seqs=[]
    for vid,frames in vids.items():
        frames=sorted(frames,key=lambda x:x[0])
        if len(frames)<3: continue
        centers=np.linspace(1,len(frames)-2,num=min(max_sequences,max(1,len(frames)-2)),dtype=int)
        for c in sorted(set(centers.tolist())):
            seqs.append((frames[c-1],frames[c],frames[c+1]))
    return seqs


class SeqDataset(Dataset):
    def __init__(self,seqs,train=False,size=224):
        self.seqs=seqs
        ops=[transforms.Resize((size,size))]
        if train: ops += [transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(0.15,0.15,0.1,0.02)]
        ops += [transforms.ToTensor(),transforms.Normalize(MEAN.flatten().tolist(),STD.flatten().tolist())]
        self.tf=transforms.Compose(ops)
    def __len__(self): return len(self.seqs)
    def __getitem__(self,i):
        triple=self.seqs[i]; xs=[]
        for f in triple:
            with Image.open(f[0]) as im: xs.append(self.tf(im.convert("RGB")))
        center=triple[1]
        return torch.stack(xs),center[1],center[3],center[4],center[2]


def sobel_magnitude(x):
    gray=(x*STD.to(x.device)+MEAN.to(x.device)).clamp(0,1)
    gray=0.2989*gray[:,0:1]+0.5870*gray[:,1:2]+0.1140*gray[:,2:3]
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device=x.device,dtype=x.dtype).view(1,1,3,3)
    ky=kx.transpose(2,3)
    gx=F.conv2d(gray,kx,padding=1); gy=F.conv2d(gray,ky,padding=1)
    return torch.sqrt(gx*gx+gy*gy+1e-6)


class TDSFInspired(nn.Module):
    def __init__(self,freeze=11):
        super().__init__()
        base=build_dinov2_fas("dinov2_vitb14_reg",pretrained=True,feature_mode="cls",freeze_first_blocks=freeze)
        self.backbone=base.backbone
        d=int(self.backbone.embed_dim)
        self.grad_branch=nn.Sequential(
            nn.Conv2d(1,32,5,2,2),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,128,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.temporal=nn.Sequential(nn.LayerNorm(d*3),nn.Linear(d*3,d),nn.GELU(),nn.Dropout(0.2))
        self.classifier=nn.Sequential(nn.LayerNorm(d+128),nn.Linear(d+128,2))
    def cls(self,x): return self.backbone.forward_features(x)["x_norm_clstoken"]
    def forward(self,seq):
        b,t,c,h,w=seq.shape
        f0=self.cls(seq[:,0]); f1=self.cls(seq[:,1]); f2=self.cls(seq[:,2])
        temporal=self.temporal(torch.cat([f1,torch.abs(f1-f0),torch.abs(f2-f1)],1))
        grad=self.grad_branch(sobel_magnitude(seq[:,1]))
        return self.classifier(torch.cat([temporal,grad],1))


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data_root",default="/root/Desktop/code/FAS/ProcessedData")
    p.add_argument("--target_dataset",required=True,choices=MICO_DATASETS)
    p.add_argument("--output_dir",default="outputs_week3_td_sf")
    p.add_argument("--eval_dir",default="outputs_week3_td_sf_eval")
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch_size",type=int,default=12)
    p.add_argument("--num_workers",type=int,default=8); p.add_argument("--freeze_first_blocks",type=int,default=11)
    p.add_argument("--backbone_lr",type=float,default=3e-6); p.add_argument("--head_lr",type=float,default=5e-5)
    p.add_argument("--max_sequences_per_video",type=int,default=10); p.add_argument("--patience",type=int,default=8)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()


def eval_video(model,loader,device):
    model.eval(); ys=[];scores=[];vids=[];domains=[]
    with torch.no_grad():
        for x,y,d,v,_ in loader:
            x=x.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                p=torch.softmax(model(x).float(),1)[:,1]
            ys+=y.tolist();scores+=p.cpu().tolist();vids+=list(v);domains+=list(d)
    _,vy,vs=aggregate_video_scores(ys,scores,vids); overall=compute_metrics(vy,vs)
    by={}
    for d in sorted(set(domains)):
        idx=[i for i,z in enumerate(domains) if z==d]
        _,dy,ds=aggregate_video_scores([ys[i] for i in idx],[scores[i] for i in idx],[vids[i] for i in idx])
        by[d]=compute_metrics(dy,ds)
    aucs=[m["auc"] for m in by.values()]
    return overall,by,min(aucs),float(np.mean(aucs))


def main():
    a=parse_args(); random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources=[d for d in MICO_DATASETS if d!=a.target_dataset]
    tr_all=[];va_all=[]
    for i,d in enumerate(sources):
        seqs=build_sequences(a.data_root,d,a.max_sequences_per_video)
        # split sequence centers by subject using generic group splitter
        pseudo=[(str(j),s[1][1],s[1][2]) for j,s in enumerate(seqs)]
        trp,vap,_=__import__("datasets").split_by_group(pseudo,seed=a.seed+17*i,train_ratio=0.9,val_ratio=0.05)
        tr_idx={int(x[0]) for x in trp}; va_idx={int(x[0]) for x in vap}
        tr_all += [s for j,s in enumerate(seqs) if j in tr_idx]
        va_all += [s for j,s in enumerate(seqs) if j in va_idx]

    counts=Counter((s[1][3],s[1][1]) for s in tr_all)
    weights=[1.0/counts[(s[1][3],s[1][1])] for s in tr_all]
    sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(weights),replacement=True)
    tr=SeqDataset(tr_all,True);va=SeqDataset(va_all,False)
    kw=dict(num_workers=a.num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=a.num_workers>0)
    tl=DataLoader(tr,batch_size=a.batch_size,sampler=sampler,drop_last=True,**kw)
    vl=DataLoader(va,batch_size=max(16,a.batch_size),shuffle=False,**kw)

    model=TDSFInspired(a.freeze_first_blocks).to(device)
    bb=[p for n,p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    hd=[p for n,p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    opt=AdamW([{"params":bb,"lr":a.backbone_lr},{"params":hd,"lr":a.head_lr}],weight_decay=0.05)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    out=Path(a.output_dir)/a.target_dataset;out.mkdir(parents=True,exist_ok=True)
    best_w=-1.;best_m=-1.;bad=0;history=[]
    for e in range(1,a.epochs+1):
        model.train();loss_sum=0.;n=0
        for x,y,_,_,_ in tl:
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits=model(x);loss=F.cross_entropy(logits,y)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update()
            loss_sum+=loss.item()*x.size(0);n+=x.size(0)
        ov,by,w,m=eval_video(model,vl,device)
        row={"epoch":e,"loss":loss_sum/n,"val_video_auc":ov["auc"],"val_worst_domain_auc":w,"val_mean_domain_auc":m,
             "val_by_domain":{d:{"auc":z["auc"],"eer":z["eer"]} for d,z in by.items()}}
        history.append(row);print(json.dumps(row,ensure_ascii=False))
        if w>best_w+1e-6 or (abs(w-best_w)<=1e-6 and m>best_m+1e-6):
            best_w,best_m=w,m;bad=0;torch.save({"model":model.state_dict(),"args":vars(a),"source_datasets":sources},out/"best.pth");print(f"[BEST] epoch={e} worst={w:.6f} mean={m:.6f}")
        else:
            bad+=1
            if bad>=a.patience: print(f"Early stopping at epoch {e}.");break
    (out/"history.json").write_text(json.dumps(history,indent=2,ensure_ascii=False),encoding="utf-8")

    model.load_state_dict(torch.load(out/"best.pth",map_location="cpu")["model"])
    target=SeqDataset(build_sequences(a.data_root,a.target_dataset,a.max_sequences_per_video),False)
    el=DataLoader(target,batch_size=32,shuffle=False,**kw)
    model.eval();ys=[];sc=[];vids=[]
    with torch.no_grad():
        for x,y,_,v,_ in el:
            x=x.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                p=torch.softmax(model(x).float(),1)[:,1]
            ys+=y.tolist();sc+=p.cpu().tolist();vids+=list(v)
    eval_out=Path(a.eval_dir)/a.target_dataset
    fm=save_plots_and_metrics(ys,sc,eval_out/"sequence_level",prefix="sequence")
    _,vy,vs=aggregate_video_scores(ys,sc,vids);vm=save_plots_and_metrics(vy,vs,eval_out/"video_level",prefix="video")
    summary={"method":"FAS-TD-SF-inspired spatial-gradient temporal model","sources":sources,"target":a.target_dataset,"sequence_metrics":fm,"video_metrics":vm,
             "note":"Inspired by FAS-TD-SF/SGTD; no PRNet virtual-depth supervision is used."}
    (eval_out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"FINAL Sequence AUC={fm['auc']:.6f} VIDEO AUC={vm['auc']:.6f} VIDEO EER={vm['eer']:.6f}")


if __name__=="__main__":
    main()
