"""Fit or evaluate GAE+ V2 using explicit checkpoint and cohort paths."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import numpy as np
import torch
from gae_plus_v2 import gate_rows,search_decoder,parameter_fingerprint
from ood_utils.saelens_adapters import load_explainer,DecoderAdapter
from ood_utils.gae_plus_v2_data import load_records,validate_disjoint,validate_frozen_cohort
from ood_utils.gae_plus_v2_evaluation import aggregate,score_prompt


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def backbone_fingerprint(model):
    cfg=dict(model.cfg.to_dict())
    for key in ['device','n_devices','seed','init_weights','model_name','tokenizer_name']:
        cfg.pop(key,None)
    h=hashlib.sha256(json.dumps(cfg,sort_keys=True,default=str).encode())
    for name,value in sorted(model.state_dict().items()):
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def validate_backbone(expected,actual):
    for key in ['checkpoint_sha256','profile','model','model_revision','model_state_sha256','backbone_sha256','kind','layer']:
        if expected.get(key)!=actual.get(key):raise ValueError('Frozen backbone/explainer identity differs: '+key)
    if not expected.get('backbone_sha256'):raise ValueError('Frozen decoder has no backbone content fingerprint')


def load_model(args):
    from transformer_lens import HookedTransformer,HookedTransformerConfig
    if args.model_state:
        allowed=[np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),
                 type(np.dtype('int64')),type(np.dtype('int32'))]
        with torch.serialization.safe_globals(allowed):
            saved=torch.load(args.model_state,map_location='cpu',weights_only=True)
        cfg=dict(saved['cfg']);cfg.update(device=args.device)
        model=HookedTransformer(HookedTransformerConfig.from_dict(cfg));model.load_state_dict(saved['state_dict'],strict=True)
        return model.to(args.device).eval().requires_grad_(False),getattr(model,'tokenizer',None)
    if not args.model or not args.model_revision:
        raise ValueError('Provide --model-state or --model and an explicit --model-revision')
    from transformers import AutoModelForCausalLM,AutoTokenizer
    hf_name='EleutherAI/pythia-1.4b' if args.model=='pythia-1.4b' else 'gpt2'
    location=args.model_dir or hf_name
    options=dict(revision=args.model_revision,local_files_only=bool(args.model_dir),cache_dir=args.cache_dir)
    tokenizer=AutoTokenizer.from_pretrained(location,**options)
    hf=AutoModelForCausalLM.from_pretrained(location,torch_dtype=torch.float32,**options)
    expected_arch='gpt_neox' if args.model=='pythia-1.4b' else 'gpt2'
    if hf.config.model_type!=expected_arch:raise ValueError('Local backbone architecture differs from --model')
    if args.model_dir:
        from transformer_lens import loading_from_pretrained as loading
        local=str(args.model_dir.resolve())
        hf_cfg=hf.config.to_dict()
        if args.model=='gpt2':hf_cfg.setdefault('n_ctx',hf.config.n_positions)
        args.cache_dir.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='gae-v2-config-',dir=args.cache_dir) as temporary:
            (Path(temporary)/'config.json').write_text(json.dumps(hf_cfg))
            cfg=loading.get_pretrained_model_config(str(Path(temporary).resolve()),hf_cfg=hf_cfg,fold_ln=True,
                device=args.device,dtype=torch.float32,local_files_only=True)
        if args.model=='gpt2' and hf.config.n_inner is not None:cfg.d_mlp=hf.config.n_inner
        model=HookedTransformer(cfg,tokenizer=tokenizer,move_to_device=False)
        state=loading.get_pretrained_state_dict(local,cfg,hf_model=hf,dtype=torch.float32)
        model.load_and_process_state_dict(state)
        model.to(args.device)
    else:
        model=HookedTransformer.from_pretrained(args.model,hf_model=hf,tokenizer=tokenizer,
            revision=args.model_revision,device=args.device,dtype=torch.float32,cache_dir=args.cache_dir)
    return model.eval().requires_grad_(False),tokenizer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['fit','evaluate'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--model',choices=['gpt2','pythia-1.4b'])
    parser.add_argument('--model-revision')
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--model-state',type=Path,help='Local TransformerLens cfg/state_dict checkpoint')
    parser.add_argument('--cache-dir',type=Path,default=Path('.cache'))
    parser.add_argument('--kind',choices=['sae','transcoder'])
    parser.add_argument('--layer',type=int,default=None)
    parser.add_argument('--profile',choices=['medical','standard'],default='medical')
    parser.add_argument('--dataset',default='custom')
    parser.add_argument('--adaptation',type=Path)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--decoder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--n-fit',type=int,default=128)
    parser.add_argument('--ranks',default='8,32,64,96')
    parser.add_argument('--lambdas',default='.001,.005,.05,.2')
    parser.add_argument('--geom',type=float,default=.1)
    parser.add_argument('--preset-rank',type=int,default=32)
    parser.add_argument('--preset-lambda',type=float,default=.2)
    parser.add_argument('--geometry',choices=['centered','second-moment'])
    parser.add_argument('--reference-solver',choices=['svd','eigh'],default=None)
    parser.add_argument('--warm-eta',help='Optional existing GAE mixing coordinates, e.g. 0.375,0.625')
    args=parser.parse_args(argv)
    if args.action=='fit' and not args.adaptation:parser.error('fit requires --adaptation')
    if args.action=='evaluate' and (not args.evaluation or not args.decoder):parser.error('evaluate requires --evaluation and --decoder')
    args.output.mkdir(parents=True,exist_ok=False)
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if torch.cuda.is_available():torch.cuda.manual_seed_all(2026)
    model,tokenizer=load_model(args);base=load_explainer(args.checkpoint,args.kind,args.device)
    layer=args.layer if args.layer is not None else 8 if args.model=='gpt2' else 15 if args.model=='pythia-1.4b' else 0
    versions=[(p,p._version) for p in list(model.parameters())+list(base.buffers())]
    checkpoint_sha=checksum(args.checkpoint)
    identity=dict(checkpoint_sha256=checkpoint_sha,model=args.model,model_revision=args.model_revision,
        model_state_sha256=checksum(args.model_state) if args.model_state else None,
        backbone_sha256=backbone_fingerprint(model),seed=2026,
        profile=args.profile,kind=base.kind,layer=layer,device=args.device,torch=str(torch.__version__))
    evaluation=load_records(args.evaluation,'evaluation',tokenizer,args.dataset,args.profile) if args.evaluation else None
    if args.action=='fit':
        records=load_records(args.adaptation,'adaptation',tokenizer,args.dataset,args.profile)
        if args.n_fit<2 or args.n_fit>len(records):raise ValueError('n-fit must use at least two available adaptation records')
        if evaluation:validate_disjoint(records,evaluation)
        records=records[:args.n_fit];features=[];targets=[]
        def hook(name):return name if name.startswith('blocks.') else f'blocks.{layer}.'+name
        input_hook,output_hook=hook(base.input_hook_point),hook(base.output_hook_point)
        with torch.inference_mode():
            for record in records:
                _,cache=model.run_with_cache(torch.tensor([record['tokens']],device=args.device),names_filter=list({input_hook,output_hook}),return_type=None)
                _,z=base(cache[input_hook][:,-1],return_z=True)
                features.append(z.cpu());targets.append(cache[output_hook][:,-1].cpu())
        measure_cache={}
        def measure(candidate,subset):
            adapter=DecoderAdapter(base,candidate['weight'].to(args.device),candidate['bias'].to(args.device))
            answer=[]
            for record in subset:
                key=(candidate['model_id'],record['content_token_sha256'])
                if key not in measure_cache:measure_cache[key]=score_prompt(model,adapter,record,layer,args.profile)
                answer.append(measure_cache[key])
            return answer
        ranks=[int(x) for x in args.ranks.split(',')];lambdas=[float(x) for x in args.lambdas.split(',')]
        recipes=[dict(rank=r,lam=lam,geom=args.geom,n_fit=args.n_fit) for r in ranks for lam in lambdas]
        if args.warm_eta:
            eta=[float(x) for x in args.warm_eta.split(',')]
            if len(eta)!=2 or min(eta)<0 or sum(eta)>1:raise ValueError('Invalid warm eta')
            for recipe in recipes:recipe['warm_eta']=eta
        reference_solver=args.reference_solver or ('eigh' if args.model=='pythia-1.4b' and args.profile=='standard' else 'svd')
        result=search_decoder(base.W_dec.T,base.b_dec_out if base.kind=='transcoder' else base.b_dec,
            torch.cat(targets),torch.cat(features),records,measure,recipes,profile=args.profile,
            preset_rank=args.preset_rank,preset_lambda=args.preset_lambda,geometry=args.geometry,reference_solver=reference_solver)
        adaptation_identity=[{k:r[k] for k in ['leakage_group','text_sha256','token_hash','content_token_sha256']} for r in records]
        torch.save(dict(weight=result['weight'],bias=result['bias'],metadata=result['metadata'],identity=identity,
                        adaptation_identity=adaptation_identity),args.output/'selected_decoder.pt')
        write(args.output/'selection.json',dict(metadata=result['metadata'],history=result['history'],fitting=result['fitting'],identity=identity))
        selected=result
    else:
        selected=torch.load(args.decoder,map_location='cpu',weights_only=True)
        validate_frozen_cohort(selected.get('adaptation_identity'),evaluation)
        validate_backbone(selected['identity'],identity)
    if evaluation:
        adapter=DecoderAdapter(base,selected['weight'].to(args.device),selected['bias'].to(args.device))
        adapted=[score_prompt(model,adapter,r,layer,args.profile) for r in evaluation]
        fixed=[score_prompt(model,base,r,layer,args.profile) for r in evaluation]
        routed=gate_rows(adapted,fixed)
        with (args.output/'raw.jsonl').open('x') as f:
            for row in routed:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        report=aggregate(routed,args.profile);report.update(identity=identity,selected=selected['metadata'],
            fallback_count=sum(r['gate_route']=='fixed' for r in routed),raw_sha256=checksum(args.output/'raw.jsonl'))
        write(args.output/'evaluation.json',report)
        print(json.dumps(report,ensure_ascii=False,allow_nan=False))
    if any(p._version!=v for p,v in versions):raise ValueError('Frozen backbone or encoder was modified')
    write(args.output/'completed.json',dict(status='completed',finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),identity=identity,frozen_parameters_unchanged=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
