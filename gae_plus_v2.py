"""GAE+ V2: frozen encoder, decoder-only refit and same-S empirical selection."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
import torch
from ood_utils.gae_plus_v2_evaluation import M_VALUES,summarize,summarize_public_scores

MEDICAL_VERSION='gae_public_c6_medical_v1'
STANDARD_VERSION='gae_public_c6_halueval_v1'
IDENTITY=['record_id','leakage_group','text_sha256']

def fit_fixed_bias(z,y,prior,bias,u,lam,geom=.1):
    """Minimize sum squared residual + lam||W-prior||² + geom||(I-UU')W||².

    The intercept is exactly bias. Z is not centered; the supplied U can be
    estimated from centered Y. This is a sum-loss ridge convention.
    """
    if not math.isfinite(lam) or lam<=0 or not math.isfinite(geom) or geom<0:
        raise ValueError('Invalid ridge strengths')
    if z.ndim!=2 or y.ndim!=2 or prior.shape!=(y.shape[1],z.shape[1]) or bias.shape!=(y.shape[1],) or len(z)!=len(y) or not len(z):
        raise ValueError('Invalid fitting shapes')
    if u.ndim!=2 or u.shape[0]!=y.shape[1] or not 0<u.shape[1]<=u.shape[0]:raise ValueError('Invalid geometry basis')
    if any(not torch.isfinite(t).all() for t in [z,y,prior,bias,u]):raise ValueError('Nonfinite fitting input')
    z,y,prior,bias,u=[t.to(dtype=torch.float64) for t in [z,y,prior,bias,u]]
    if not torch.allclose(u.T@u,torch.eye(u.shape[1],dtype=u.dtype,device=u.device),atol=1e-6,rtol=1e-6):raise ValueError('Geometry basis is not orthonormal')
    inside=u@(u.T@prior);outside=prior-inside
    effective_prior=inside+(lam/(lam+geom))*outside
    del inside,outside
    residual=y-bias-z@effective_prior.T
    inside_residual=(residual@u)@u.T;outside_residual=residual-inside_residual
    gram=z@z.T;gram=(gram+gram.T)*.5;identity=torch.eye(len(z),dtype=z.dtype,device=z.device)
    alpha=torch.linalg.solve(gram+lam*identity,inside_residual)+torch.linalg.solve(gram+(lam+geom)*identity,outside_residual)
    del gram,identity,inside_residual,outside_residual,residual
    weight=effective_prior+alpha.T@z
    del effective_prior,alpha
    residual_after=z@weight.T+bias-y;outside_after=weight-u@(u.T@weight)
    gradient=residual_after.T@z+lam*(weight-prior)+geom*outside_after
    del outside_after
    scale=((y-bias).T@z+lam*prior).norm().clamp_min(1e-12)
    info=dict(fit_samples=len(z),rank=u.shape[1],lam=lam,geom=geom,centering='Z and residual target uncentered; supplied geometry basis separately defined',
        ridge_convention='sum-loss; no division by N',solver_dtype='float64',stationarity_relative_residual=float(gradient.norm()/scale),
        train_mean_coordinate_mse=float(residual_after.square().mean()),objective='sum ||Y-ZW^T-b0||² + lam||W-W1||² + geom||(I-UU^T)W||²; b=b0')
    if not torch.isfinite(weight).all():raise ValueError('Nonfinite fitted decoder')
    return weight,bias.clone(),info

def eta_grid(warm=None):
    values=[[0.,0.],[0.,.5],[0.,1.],[.5,0.],[.5,.5],[1.,0.]]
    if warm is not None and warm not in values:values.append(list(warm))
    return values

def mix(endpoints,eta):
    a,b=map(float,eta)
    if min(a,b)<0 or a+b>1+1e-12:raise ValueError('Invalid simplex coordinates')
    return ((1-a-b)*endpoints[0]+a*endpoints[1]+b*endpoints[2]).float()

def refit(z,y,prior,bias,u,lam,geom,fixed_bias):
    u=u.double()
    before=float((u.T@u-torch.eye(u.shape[1],device=u.device,dtype=u.dtype)).abs().max())
    # Preserve the estimated span, restoring the projector assumption in FP64.
    repaired=before>1e-12
    if repaired:u=torch.linalg.qr(u,mode='reduced').Q
    geometry=dict(geometry_fp64_reorthogonalized=repaired,geometry_gram_max_error_before=before,
                  geometry_gram_max_error_after=float((u.T@u-torch.eye(u.shape[1],device=u.device,dtype=u.dtype)).abs().max()))
    if fixed_bias:
        w,b,info=fit_fixed_bias(z,y,prior,bias,u,lam,geom)
        return w,b,dict(info,**geometry)
    z,y=z.double(),y.double();mz,my=z.mean(0),y.mean(0)
    w,_,info=fit_fixed_bias(z-mz,y-my,prior,torch.zeros_like(my),u,lam,geom)
    return w,my-w@mz,dict(info,bias='free_affine_intercept',**geometry)

def step1(base,y,rank,bases=None):
    if not 0<rank<=min(base.shape[0],len(y)-1):raise ValueError('Centered rank exceeds sample/dimension rank')
    if bases is None:
        reference,_,_=torch.linalg.svd(base,full_matrices=False)
        centered=y-y.mean(0)
        values,ood=torch.linalg.eigh(centered.T@centered/max(len(y)-1,1))
        bases=(reference,values,ood)
    reference,values,ood=bases
    if values[-rank]<=max(float(values[-1])*1e-10,1e-15):raise ValueError('Requested geometry is numerically rank deficient')
    u=ood[:,-rank:].flip(1);v=reference[:,:rank]
    left,_,right=torch.linalg.svd(u.T@v,full_matrices=False)
    weight=u@(left@right)@(v.T@base)
    weight=weight*(base.norm(dim=0)/(weight.norm(dim=0)+1e-8))
    return weight,u,bases

def standard_step1(weight,u_id,y,rank,match_norms=True):
    if rank>min(len(y),y.shape[1]):raise ValueError('Requested geometry rank exceeds available data')
    _,_,vh=torch.linalg.svd(y.double(),full_matrices=False)
    u=vh[:rank].T
    ref=u_id[:,:rank].double()
    left,_,right=torch.linalg.svd(u.T@ref,full_matrices=False)
    w=u@(left@right)@(ref.T@weight.double())
    if match_norms:w=w*(weight.double().norm(dim=0)/(w.norm(dim=0)+1e-8))
    return w,u

def aligned(a,b):
    if len(a)!=len(b) or any(tuple(x[k] for k in IDENTITY)!=tuple(y[k] for k in IDENTITY) for x,y in zip(a,b)):
        raise ValueError('Selection input identity/order differs')

def gate_rows(adapted,fixed):
    aligned(adapted,fixed);out=[]
    for a,f in zip(adapted,fixed):
        accepted=a['s0']-a['s_empty']>.1
        out.append(dict(a if accepted else f,gate_route='adapted' if accepted else 'fixed',
                        gate_candidate_margin=a['s0']-a['s_empty']))
    return out

def select_on_s(candidates,reference_ids,eligible_ids=None):
    if not candidates or len({c['id'] for c in candidates})!=len(candidates):raise ValueError('Empty or duplicate candidates')
    lookup={c['id']:i for i,c in enumerate(candidates)}
    fixed=[c for c in candidates if c.get('fixed')]
    if len(fixed)!=1:raise ValueError('Exactly one Fixed reference is required')
    fixed=fixed[0]
    if not reference_ids or fixed['id'] not in reference_ids or set(reference_ids)-lookup.keys():raise ValueError('Invalid selection references')
    eligible=set(lookup) if eligible_ids is None else set(eligible_ids)
    if not eligible or eligible-lookup.keys():raise ValueError('Invalid eligible candidates')
    values=[];valid=[]
    for candidate in candidates:
        aligned(candidate['rows'],fixed['rows']);a=[];mask=[]
        for raw,ref in zip(candidate['rows'],fixed['rows']):
            if raw.get('role')!='adaptation' or ref.get('role')!='adaptation':raise ValueError('Internal selection requires adaptation S only')
            if raw.get('metric_version')!=MEDICAL_VERSION or ref.get('metric_version')!=MEDICAL_VERSION:raise ValueError('Medical metric profile differs')
            margin=ref['s0']-ref['s_empty']
            surrogate=summarize_public_scores(dict(raw,s_empty=raw['s0']-margin))
            own=summarize_public_scores(raw)
            a.append([surrogate['n_aopc'],surrogate['n_comp']]);mask.append(own['norm_valid'] and surrogate['norm_valid'])
        values.append(a);valid.append(mask)
    mask=np.all(valid,axis=0)
    if not mask.any():raise ValueError('No mathematically defined common selection scores')
    array=np.asarray(values,dtype=float)[:,mask,:];references=[lookup[c] for c in reference_ids]
    means=array.mean(1);reference=means[references].max(0)
    rng=np.random.default_rng(2026);index=rng.integers(0,int(mask.sum()),size=(500,int(mask.sum())))
    boot=np.stack([v[index].mean(1) for v in array]);br=boot[references].max(0)
    scores=np.quantile(((boot-br)/np.maximum(abs(br),.1)).min(2),.2,axis=1)
    ranked=[]
    for i,c in enumerate(candidates):
        if c['id'] not in eligible:continue
        eta=c.get('eta',[0.,0.])
        ranked.append(dict(id=c['id'],score=float(scores[i]),n_aopc_surrogate=float(means[i,0]),
            n_comp_surrogate=float(means[i,1]),eta=eta,fixed=bool(c.get('fixed')),recipe=c.get('recipe',{})))
    ranked.sort(key=lambda r:(-r['score'],sum(r['eta']),r['eta'][1],r['recipe'].get('rank',0),r['recipe'].get('lam',0),r['id']))
    return dict(selected_id=ranked[0]['id'],ranked=ranked,common_valid=int(mask.sum()),
        total=len(mask),bootstrap_repeats=500,bootstrap_quantile=.2,reference_ids=reference_ids,
        profile='fixed_reference_denominator_surrogate_v1',valid_fraction_filter=False)

def promoted_ids(boards,stage,reference_ids):
    keep=set(reference_ids)
    for family,count in [('main',3 if stage==0 else 1),('endpoint',2 if stage==0 else 1),('original_endpoint',3 if stage==0 else 1)]:
        chosen=0
        for row in boards[family]['ranked']:
            if row.get('fixed'):continue
            keep.add(row['id']);chosen+=1
            if chosen==count:break
    return keep

class InsufficientSelectionEvidence(ValueError):
    """No variability can be estimated; advance within S or abstain at full S."""
    def __init__(self,effective,total):
        self.effective=int(effective);self.total=int(total)
        super().__init__(f'Selection evidence insufficient: {effective}/{total} common-valid rows')

def _tie(c):
    eta=c.get('eta',[0.,0.]);sig=c.get('signature') or {}
    return (sum(eta),eta[1],sig.get('rank',0),sig.get('lam_gae',0),c['id'])

def select_causal(candidates, mode, seed=2026, exclude_low_valid=False, reference_ids=None, normalizer_id=None,eligible_ids=None,validity_policy='no_ratio_v1'):
    if not candidates or mode not in ['empirical','bootstrap_q20']:raise ValueError('Invalid selector request')
    if validity_policy!='no_ratio_v1':raise ValueError('Unknown selection validity policy')
    if len({c['id'] for c in candidates})!=len(candidates):raise ValueError('Repeated candidate id')
    eligible={c['id'] for c in candidates} if eligible_ids is None else set(eligible_ids)
    if not eligible or eligible-{c['id'] for c in candidates}:raise ValueError('Unknown or empty eligible selection set')
    fixed=[i for i,c in enumerate(candidates) if c.get('fixed')]
    if len(fixed)!=1:raise ValueError('Exactly one Fixed reference required')
    normalization_rows=None
    if normalizer_id is not None:
        reference=next((c for c in candidates if c['id']==normalizer_id and c.get('fixed')),None)
        if reference is None:raise ValueError('Fixed normalization reference is missing')
        normalization_rows=reference['rows']
    identities=None;values=[];valid=[];kinds=None;targets=None
    for c in candidates:
        if c.get('role')!='adaptation':raise ValueError('Selection requires adaptation S only')
        rows=c['rows'];ids=[(r['group_id'],r['token_hash']) for r in rows]
        if not rows or len({g for g,t in ids})!=len(rows) or len({t for g,t in ids})!=len(rows):raise ValueError('Empty or duplicate selection input')
        if identities is not None and ids!=identities:raise ValueError('Selection input identities/order differ')
        identities=ids
        this_targets=[r.get('target_argmax') for r in rows]
        if targets is not None and targets!=this_targets:raise ValueError('Selection target argmax differs')
        targets=this_targets
        this_kinds=[r['kind'] for r in rows]
        if kinds is not None and kinds!=this_kinds:raise ValueError('Input kinds differ')
        if set(this_kinds)-{'qa','dialogue','fineweb','edgar'}:raise ValueError('Unknown input kind')
        kinds=this_kinds;v=[]
        for position,row in enumerate(rows):
            if row.get('measurement_role','adaptation')!='adaptation':raise ValueError('Selection raw is not adaptation S')
            if row.get('metric_version')!=STANDARD_VERSION or row.get('m_values')!=M_VALUES:raise ValueError('Metric profile differs')
            value=summarize(row['s0'],row['s_empty'],row['s_del'])
            if normalization_rows is not None:
                reference=normalization_rows[position]
                gap=reference['s0']-reference['s_empty']
                # Internal selection surrogate only. Original raw and reporting
                # denominators are never overwritten or relabeled.
                surrogate=summarize(row['s0'],row['s0']-gap,row['s_del'])
                value=dict(value,n_aopc=surrogate['n_aopc'],n_comp=surrogate['n_comp'],
                    norm_valid=value['norm_valid'] and surrogate['norm_valid'])
            v.append(value)
        values.append([[r['n_aopc'],r['n_comp']] for r in v]);valid.append([r['norm_valid'] for r in v])
    rejected=[];filter_low_valid=False
    mask=np.all(valid,axis=0)
    if validity_policy=='no_ratio_v1' and mask.sum()<2:
        raise InsufficientSelectionEvidence(mask.sum(),len(mask))
    array=np.asarray(values,dtype=float)[:,mask,:];means=array.mean(1);f=fixed[0]
    reference_ids=[candidates[f]['id']] if reference_ids is None else list(reference_ids)
    if not reference_ids or candidates[f]['id'] not in reference_ids or len(reference_ids)!=len(set(reference_ids)):
        raise ValueError('References must uniquely include Fixed')
    lookup={c['id']:i for i,c in enumerate(candidates)}
    if any(x not in lookup for x in reference_ids):raise ValueError('Missing selection reference')
    references=[lookup[x] for x in reference_ids];reference=means[references].max(0)
    scale=np.maximum(np.abs(reference),.1);utility=((means-reference)/scale).min(1)
    if mode=='bootstrap_q20':
        rng=np.random.default_rng(seed);k=np.asarray(kinds)[mask]
        draws=[]
        for kind in ['qa','dialogue','fineweb','edgar']:
            index=np.flatnonzero(k==kind)
            if len(index):draws.append(rng.choice(index,size=(500,len(index)),replace=True))
        index=np.concatenate(draws,axis=1)
        boot=np.stack([v[index].mean(1) for v in array])
        boot_reference=boot[references].max(0)
        relative=(boot-boot_reference)/np.maximum(np.abs(boot_reference),.1)
        scores=np.quantile(relative.min(2),.2,axis=1)
    else:scores=utility
    ranked=[]
    for i,c in enumerate(candidates):
        if c['id'] not in eligible:continue
        ranked.append(dict(id=c['id'],model_id=c.get('model_id'),eta=c.get('eta',[0.,0.]),signature=c.get('signature'),fixed=bool(c.get('fixed')),
            n_aopc=float(means[i,0]),n_comp=float(means[i,1]),utility=float(utility[i]),selection_score=float(scores[i]),own_valid=int(sum(valid[i]))))
    if not ranked:raise ValueError('No feasible eligible selection candidate')
    ranked.sort(key=lambda r:(-r['selection_score'],*_tie(r)))
    return dict(selected_id=ranked[0]['id'],ranked=ranked,mode=mode,common_valid=int(mask.sum()),total=len(mask),
        common_valid_ids=[g for (g,t),ok in zip(identities,mask) if ok],metric_version=STANDARD_VERSION,
        candidate_validity_filter=bool(filter_low_valid),rejected_candidates=rejected,validity_policy=validity_policy,
        reference_ids=reference_ids,reference_envelope=dict(n_aopc=float(reference[0]),n_comp=float(reference[1])),
        normalizer_id=normalizer_id,selection_metric_profile='fixed_reference_denominator_surrogate_v1' if normalizer_id is not None else STANDARD_VERSION,
        eligible_ids=sorted(eligible),
        bootstrap_repeats=500 if mode=='bootstrap_q20' else 0,bootstrap_quantile=.2 if mode=='bootstrap_q20' else None,
        interpretation='Empirical S-only selection; same-S fitting bias is not corrected and no population confidence bound is claimed.')


def parameter_fingerprint(weight,bias):
    h=hashlib.sha256()
    for name,value in [('weight',weight),('bias',bias)]:
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _description(candidate):return {k:v for k,v in candidate.items() if k not in ['weight','bias','rows']}


@torch.inference_mode()
def search_decoder(w0,b0,y,z,records,measure,recipes,profile='medical',preset_rank=32,
                   preset_lambda=.2,geometry=None,reference_solver='svd'):
    """Fit on S and select using only S; measure(candidate, records) returns raw logits.

    Recipes contain rank, lam, geom and n_fit in the sum-loss ridge convention.
    Prediction tensors are held on CPU between measurements.
    """
    n=len(records)
    if profile not in ['medical','standard'] or not n or len(y)!=n or len(z)!=n:
        raise ValueError('Fitting tensors and adaptation cohort must have identical lengths')
    if any(r.get('role',r.get('measurement_role'))!='adaptation' for r in records):
        raise ValueError('Decoder search requires adaptation records only')
    if y.ndim!=2 or z.ndim!=2 or w0.shape!=(y.shape[1],z.shape[1]) or b0.shape!=(y.shape[1],):
        raise ValueError('Invalid decoder fitting dimensions')
    if geometry is None:geometry='centered' if profile=='medical' else 'second-moment'
    if geometry not in ['centered','second-moment']:raise ValueError('Unknown geometry')
    if reference_solver not in ['svd','eigh']:raise ValueError('Unknown reference basis solver')
    recipes=[dict(r) for r in recipes]
    for recipe in recipes:
        if recipe.get('n_fit',n)!=n:raise ValueError('All recipes must use the same S')
        recipe.update(n_fit=n,geom=recipe.get('geom',.1))
        recipe.setdefault('id',hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:16])
    preset=next((r for r in recipes if r['rank']==preset_rank and r['lam']==preset_lambda and r['geom']==.1),None)
    if preset is None:raise ValueError('Recipe grid must include the rank/lambda preset with geom=0.1')
    device=w0.device;y=y.to(device);geometry_cache={};fit_info=[];measurement_cache={}
    u_id=None
    if geometry=='second-moment':
        if reference_solver=='eigh':u_id=torch.linalg.eigh(w0.double()@w0.double().T).eigenvectors.flip(1)
        else:u_id=torch.linalg.svd(w0,full_matrices=False).U
    def case(weight,bias,family,recipe=None,eta=(0.,0.),fixed=False):
        weight=weight.detach().float().cpu().contiguous();bias=bias.detach().float().cpu().contiguous()
        identity=parameter_fingerprint(weight,bias)
        if profile=='standard':identifier=identity[:24]
        elif fixed:identifier='fixed'
        elif family=='original':identifier='original_'+recipe['id']
        else:identifier='modified_'+hashlib.sha256(json.dumps([recipe,list(eta)],sort_keys=True).encode()).hexdigest()[:16]
        return dict(id=identifier,model_id=identity,weight=weight,bias=bias,family=family,recipe=recipe or {},
            eta=list(eta),fixed=fixed,signature=dict(rank=recipe['rank'],lam_gae=recipe['lam']) if recipe else None,role='adaptation')
    fixed=case(w0,b0,'fixed',fixed=True);originals=[fixed]
    def endpoints(recipe):
        rank=recipe['rank']
        if rank not in geometry_cache:
            if geometry=='centered':prior,u,_=step1(w0,y,rank)
            else:prior,u=standard_step1(w0,u_id,y,rank)
            geometry_cache[rank]=(prior.cpu(),u.cpu())
        return tuple(t.to(device) for t in geometry_cache[rank])
    for recipe in recipes:
        try:prior,u=endpoints(recipe)
        except ValueError as error:
            if 'rank deficient' in str(error):continue
            raise
        features=z.to(device=device,dtype=torch.float64) if n>=4096 else z.to(device)
        weight,bias,info=refit(features,y,prior,b0,u,recipe['lam'],recipe['geom'],False)
        originals.append(case(weight,bias,'original',recipe,(0.,1.)))
        fit_info.append(dict(recipe=recipe,family='original',info=info))
        del features,weight,bias
    preset_case=next((c for c in originals if c['recipe'].get('id')==preset['id']),None)
    if preset_case is None:raise ValueError('Preset geometry is numerically infeasible')
    references=[fixed['id'],preset_case['id']]
    def measured(cases,count):
        answer=[]
        fixed_key=(fixed['model_id'],count)
        if fixed_key not in measurement_cache:
            measurement_cache[fixed_key]=measure(fixed,records[:count])
        for c in cases:
            key=(c['model_id'],count)
            if key not in measurement_cache:
                raw=measure(c,records[:count])
                if len(raw)!=count:raise ValueError('Measurement callback returned incomplete S')
                measurement_cache[key]=raw
            routed=gate_rows(measurement_cache[key],measurement_cache[fixed_key]) if not c['fixed'] else measurement_cache[key]
            answer.append(dict(c,rows=routed))
        return answer
    def select(pool,eligible=None):
        if profile=='medical':return select_on_s(pool,references,eligible)
        return select_causal(pool,'bootstrap_q20',reference_ids=references,normalizer_id=fixed['id'],eligible_ids=eligible,validity_policy='no_ratio_v1')
    def try_selection(cases,count,operation):
        while True:
            pool=measured(cases,count)
            try:return operation(pool),pool,count
            except (InsufficientSelectionEvidence,ValueError) as error:
                missing=isinstance(error,InsufficientSelectionEvidence) or 'No mathematically defined' in str(error)
                if not missing:raise
                if count==n:return None,pool,count
                count=min(128,n) if count<min(128,n) else n
    screen,_,count=try_selection(originals,min(32,n),select)
    history=[];selected=fixed;cases=list(originals)
    if screen is not None:
        shortlist=[];ranks=set()
        for item in screen['ranked']:
            c=next(c for c in originals if c['id']==item['id'])
            if c['fixed'] or c['recipe']['rank'] in ranks:continue
            shortlist.append(c);ranks.add(c['recipe']['rank'])
            if len(shortlist)==2:break
        if len(shortlist)<2:raise ValueError('Two distinct feasible ranks are required for V2 screening')
        for original in shortlist:
            recipe=original['recipe'];prior,u=endpoints(recipe)
            features=z.to(device=device,dtype=torch.float64) if n>=4096 else z.to(device)
            weight,bias,info=refit(features,y,prior,b0,u,recipe['lam'],recipe['geom'],True)
            fit_info.append(dict(recipe=recipe,family='fixed_bias',info=info))
            for eta in eta_grid(recipe.get('warm_eta')):
                if eta==[0.,0.]:continue
                if profile=='standard':mixed=(1-sum(eta))*w0.double()+eta[0]*prior+eta[1]*weight
                else:mixed=mix([w0.float(),prior.float(),weight.float()],eta)
                c=case(mixed,b0,'fixed_bias',recipe,eta)
                if profile=='medical' or all(old['id']!=c['id'] for old in cases):cases.append(c)
            del features,weight,bias
        def boards_for(pool):
            eligible={
                'main':[c['id'] for c in pool if c['fixed'] or c['family']=='fixed_bias'],
                'endpoint':[c['id'] for c in pool if c['fixed'] or c['family']=='fixed_bias' and c['eta']==[0.,1.]],
                'original_endpoint':[c['id'] for c in pool if c['fixed'] or c['family']=='original']}
            result={}
            for family,ids in eligible.items():
                subset=pool if profile=='medical' else [c for c in pool if c['id'] in set(ids)|set(references)]
                result[family]=select(subset,ids)
            return result
        boards,pool,count=try_selection(cases,count,boards_for)
        history.append(dict(count=count,boards=boards))
        stages=list(dict.fromkeys([min(128,n),n])) if profile=='medical' else [min(128,n),n]
        stage_index=0
        for wanted in stages:
            if profile=='medical' and wanted<=count:continue
            if boards is None:break
            keep=promoted_ids(boards,stage_index,references)
            cases=[c for c in cases if c['id'] in keep]
            boards,pool,count=try_selection(cases,max(count,wanted),boards_for)
            history.append(dict(count=count,boards=boards));stage_index+=1
        if boards is not None:
            eligible=set(references)|{boards['main']['selected_id'],boards['original_endpoint']['selected_id']}
            final_pool=pool if profile=='medical' else [c for c in pool if c['id'] in eligible]
            final=select(final_pool,sorted(eligible))
            selected=next(c for c in cases if c['id']==final['selected_id'])
    metadata=_description(selected)
    metadata.update(profile=profile,geometry=geometry,reference_solver=reference_solver,n_fit=n,
        selection_uses_same_S=True,evaluation_used_for_selection=False,
        selection_unavailable=screen is None,bootstrap_repeats=500,bootstrap_quantile=.2,
        inference_policy='unperturbed_target_margin_fixed_fallback_v1')
    return dict(weight=selected['weight'],bias=selected['bias'],metadata=metadata,history=history,fitting=fit_info)
