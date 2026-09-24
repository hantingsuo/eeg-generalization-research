"""Frozen, two-phase SEED checkpoint-selection sensitivity experiment.

No B trial features are loaded until every training cell has frozen its three
policies. This is a post-rejection reanalysis of previously used participants.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pcma.data.libeer_seed import resolve_seed_subject_file
from pcma.eval.protocol_audit import (audit_contact_log, earliest_best, paired_subject_interval,
                                     reporting_summary, validate_trial_partitions)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_trials(path, labels, ids):
    """Read only named whole-trial variables, never the entire MAT payload."""
    data = loadmat(path, variable_names=[f'de_LDS{i}' for i in ids])
    features, targets, trials = [], [], []
    for i in ids:
        raw = np.asarray(data[f'de_LDS{i}'])
        if raw.ndim != 3 or raw.shape[0] != 62 or raw.shape[2] != 5:
            raise ValueError('unexpected feature shape')
        x = np.ascontiguousarray(raw.transpose(1, 0, 2), dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError('non-finite features')
        features.append(x)
        targets.append(np.full(len(x), labels[i-1], dtype=np.int64))
        trials.append(np.full(len(x), i, dtype=np.int16))
    return np.concatenate(features), np.concatenate(targets), np.concatenate(trials)


def load_model_module(cfg):
    vendor = ROOT / cfg['vendor_root']
    source, config = vendor/'models/DGCNN.py', vendor/'config/model_param/DGCNN.yaml'
    if sha(source) != cfg['dgcnn_source_sha256'] or sha(config) != cfg['dgcnn_config_sha256']:
        raise RuntimeError('pinned model/config hash mismatch')
    spec = importlib.util.spec_from_file_location('_revision_pinned_dgcnn', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.param_path = str(config)
    return module


def make_model(name, cfg, dgcnn):
    if name == 'dgcnn':
        with contextlib.redirect_stdout(io.StringIO()):
            model = dgcnn.DGCNN(num_electrodes=62, in_channels=5, num_classes=3)
        return model
    if name != 'mlp':
        raise ValueError('unknown model')
    a, b = cfg['mlp']['hidden']
    dropout = cfg['mlp']['dropout']
    return nn.Sequential(nn.Flatten(), nn.Linear(310,a), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(a,b), nn.ReLU(), nn.Dropout(dropout), nn.Linear(b,3))


@torch.no_grad()
def predict(model, features, batch_size, device):
    """Manual slices avoid DataLoader iterator RNG contacts during scoring."""
    model.eval()
    return np.concatenate([model(torch.from_numpy(features[i:i+batch_size]).to(device)).cpu().numpy()
                           for i in range(0,len(features),batch_size)])


def snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def summarize_predictions(logits, partition, subject):
    _, y, trials = partition
    return reporting_summary(logits,y,np.full(len(y),subject),trials)


def run_fit(cell, cfg, dgcnn, labels, root, device, record, check_time):
    model_name, session, subject, seed = cell
    key = f'{model_name}_s{session}_p{subject:02d}_r{seed}'
    dest = root/key
    dest.mkdir()
    started = time.monotonic()
    source = resolve_seed_subject_file(ROOT/cfg['feature_root'],session,subject)
    parts = {name:load_trials(source,labels,cfg['trials'][name]) for name in ('train','validation','selection')}
    seed_all(seed)
    model = make_model(model_name,cfg,dgcnn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'],eps=cfg['adam_epsilon'])
    regularizer = dgcnn.NewSparseL2Regularization(cfg['dgcnn_regularizer']).to(device) if model_name=='dgcnn' else None
    generator = torch.Generator().manual_seed(seed)
    x,y,_ = parts['train']
    loader = DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=cfg['batch_size'],
                        shuffle=True,num_workers=0,generator=generator)
    states, best, epochs, history = {}, {'validation':-1.,'selection':-1.}, {}, []
    for epoch in range(1,cfg['epochs']+1):
        check_time()
        model.train()
        total_loss=0.
        for bx,by in loader:
            optimizer.zero_grad()
            logits=model(bx.to(device))
            loss=nn.functional.cross_entropy(logits,by.to(device))
            if regularizer is not None:
                loss=loss+regularizer(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{key} epoch {epoch}: nonfinite loss')
            loss.backward()
            optimizer.step()
            total_loss+=float(loss.detach().cpu())*len(by)
        row={'epoch':epoch,'train_loss':total_loss/len(y)}
        for name in ('validation','selection'):
            scores=predict(model,parts[name][0],cfg['evaluation_batch_size'],device)
            accuracy=float(np.mean(scores.argmax(1)==parts[name][1]))
            row[name+'_accuracy']=accuracy
            if accuracy>best[name]:
                best[name]=accuracy
                states[name]=snapshot(model)
                epochs[name]=epoch
        history.append(row)
    states['fixed']=snapshot(model)
    epochs['fixed']=cfg['epochs']
    for name in ('validation','selection'):
        assert epochs[name]==earliest_best([r[name+'_accuracy'] for r in history])
    payload={}
    metrics={}
    for policy in cfg['policies']:
        model.load_state_dict(states[policy])
        metrics[policy]={}
        for part in ('validation','selection'):
            logits=predict(model,parts[part][0],cfg['evaluation_batch_size'],device)
            payload[f'{policy}_{part}_logits']=logits
            metrics[policy][part]=summarize_predictions(logits,parts[part],subject)
    for part in ('validation','selection'):
        payload[f'{part}_labels']=parts[part][1]
        payload[f'{part}_trials']=parts[part][2]
    with (dest/'checkpoints.pt').open('xb') as handle:
        torch.save(states,handle)
    with (dest/'selection_predictions.npz').open('xb') as handle:
        np.savez_compressed(handle,**payload)
    write_json(dest/'fit.json',{'cell':key,'model':model_name,'session':session,'subject':subject,'seed':seed,
                              'source_file':str(source.relative_to(ROOT)),'source_sha256':sha(source),
                              'epochs':epochs,'history':history,'metrics':metrics,
                              'elapsed_seconds':time.monotonic()-started,'audit_features_loaded':False,
                              'checkpoint_sha256':sha(dest/'checkpoints.pt')})
    record('selection_frozen',key,fit_sha256=sha(dest/'fit.json'))
    return key


def run_audit(key,cfg,dgcnn,labels,root,device,record):
    dest=root/key
    fit=json.loads((dest/'fit.json').read_text(encoding='utf-8'))
    if sha(dest/'checkpoints.pt')!=fit['checkpoint_sha256'] or sha(ROOT/fit['source_file'])!=fit['source_sha256']:
        raise RuntimeError('input or checkpoint changed after freeze')
    record('audit_loaded',key)
    part=load_trials(ROOT/fit['source_file'],labels,cfg['trials']['audit'])
    states=torch.load(dest/'checkpoints.pt',map_location='cpu',weights_only=True)
    seed_all(fit['seed'])
    model=make_model(fit['model'],cfg,dgcnn).to(device)
    arrays={'audit_labels':part[1],'audit_trials':part[2]}
    metrics={}
    for policy in cfg['policies']:
        model.load_state_dict(states[policy])
        logits=predict(model,part[0],cfg['evaluation_batch_size'],device)
        arrays[f'{policy}_audit_logits']=logits
        metrics[policy]=summarize_predictions(logits,part,fit['subject'])
    with (dest/'audit_predictions.npz').open('xb') as handle:
        np.savez_compressed(handle,**arrays)
    write_json(dest/'audit.json',{'cell':key,'metrics':metrics})
    record('audit_scored',key)


def aggregate(cfg,root,keys,events):
    cells=[]
    for key in keys:
        fit=json.loads((root/key/'fit.json').read_text(encoding='utf-8'))
        audit=json.loads((root/key/'audit.json').read_text(encoding='utf-8'))
        cells.append((fit,audit))
    results={}
    for model in cfg['models']:
        subset=[(f,a) for f,a in cells if f['model']==model]
        per_subject={}
        for subject in cfg['subjects']:
            unit=[(f,a) for f,a in subset if f['subject']==subject]
            values={}
            for policy in cfg['policies']:
                for part in ('selection','audit'):
                    for metric in ('window_accuracy','trial_accuracy'):
                        vals=[(a['metrics'][policy] if part=='audit' else f['metrics'][policy][part])['per_subject'][str(subject)][metric] for f,a in unit]
                        values[f'{part}_{policy}_{metric}']=float(np.mean(vals))
            per_subject[str(subject)]=values
        stats={}
        for part in ('selection','audit'):
            for metric in ('window_accuracy','trial_accuracy'):
                for left,right in (('selection','validation'),('validation','fixed'),('selection','fixed')):
                    values=[x[f'{part}_{left}_{metric}']-x[f'{part}_{right}_{metric}'] for x in per_subject.values()]
                    stats[f'{part}_{left}_minus_{right}_{metric}']=paired_subject_interval(values,
                        replicates=cfg['uncertainty']['replicates'],seed=cfg['uncertainty']['seed'])
        diff=[x['selection_selection_window_accuracy']-x['selection_validation_window_accuracy']-
              (x['audit_selection_window_accuracy']-x['audit_validation_window_accuracy']) for x in per_subject.values()]
        stats['difference_in_differences_window_accuracy']=paired_subject_interval(diff)
        means={key:float(np.mean([x[key] for x in per_subject.values()])) for key in next(iter(per_subject.values()))}
        results[model]={'means':means,'contrasts':stats,'per_subject':per_subject,
                        'selected_epoch_mean':{p:float(np.mean([f['epochs'][p] for f,a in subset])) for p in cfg['policies']}}
    contacts=audit_contact_log(events,len(keys))
    if not contacts['valid']:
        raise RuntimeError(str(contacts))
    return {'status':'completed','protocol_id':cfg['protocol_id'],'training_cells':len(keys),
            'models':results,'contact_audit':contacts,'historical_exposure':cfg['historical_exposure'],
            'interpretation_boundary':cfg['interpretation_boundary']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'plans/2026-09-10-selection-sensitivity-v1.json')
    p.add_argument('--output',type=Path,default=ROOT/'results/revision_2026-09-10/selection_v1')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    cfg=json.loads(args.manifest.read_text(encoding='utf-8'))
    cfg_hash=sha(args.manifest)
    code_hash=sha(__file__)
    if args.output.exists():
        raise FileExistsError('refusing to overwrite experiment output')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if shutil.disk_usage(ROOT).free<4*1024**3:
        raise RuntimeError('less than 4 GiB free')
    dgcnn=load_model_module(cfg)
    labels=loadmat(ROOT/cfg['feature_root']/'label.mat',variable_names=['label'])['label'].ravel().astype(np.int64)+1
    validate_trial_partitions(cfg['trials'],labels)
    cells=list(itertools.product(cfg['models'],cfg['sessions'],cfg['subjects'],cfg['optimization_seeds']))
    if len(cells)!=cfg['expected_training_cells']:
        raise RuntimeError('unexpected experimental matrix')
    args.output.mkdir(parents=True)
    root=args.output/'cells';root.mkdir()
    shutil.copyfile(args.manifest,args.output/'frozen_manifest.json')
    write_json(args.output/'runtime.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                                        'platform':platform.platform(),'device':args.device,
                                        'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                                        'manifest_sha256':cfg_hash,'runner_sha256':code_hash,
                                        'audit_module_sha256':sha(ROOT/'pcma/eval/protocol_audit.py')})
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    started=time.monotonic()
    events=[]
    def record(event,cell=None,**extra):
        row={'time_utc':datetime.now(timezone.utc).isoformat(),'event':event,'cell':cell,**extra}
        events.append(row)
        with (args.output/'events.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
    def check_time():
        if time.monotonic()-started>cfg['maximum_wall_seconds']:
            raise TimeoutError('frozen hard runtime limit reached')
    try:
        record('run_started',manifest_sha256=cfg_hash,runner_sha256=code_hash)
        keys=[]
        for i,cell in enumerate(cells,1):
            check_time()
            key=run_fit(cell,cfg,dgcnn,labels,root,device,record,check_time)
            keys.append(key)
            print(json.dumps({'phase':'fit','completed':i,'total':len(cells),'cell':key,'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
        if sha(args.manifest)!=cfg_hash or sha(__file__)!=code_hash:
            raise RuntimeError('configuration/code changed during experiment')
        write_json(args.output/'all_selections_frozen.json',{'cells':keys,'manifest_sha256':cfg_hash,
                     'fit_sha256':{key:sha(root/key/'fit.json') for key in keys},'time_utc':datetime.now(timezone.utc).isoformat()})
        record('all_selections_frozen')
        for i,key in enumerate(keys,1):
            check_time()
            run_audit(key,cfg,dgcnn,labels,root,device,record)
            if i%15==0:
                print(json.dumps({'phase':'audit','completed':i,'total':len(keys)}),flush=True)
        result=aggregate(cfg,root,keys,events)
        result['elapsed_seconds']=time.monotonic()-started
        write_json(args.output/'summary.json',result)
        record('run_completed')
        print(json.dumps({'status':'completed','summary':str(args.output/'summary.json'),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except BaseException as exc:
        record('run_failed',error=repr(exc))
        write_json(args.output/'failure.json',{'status':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    main()
