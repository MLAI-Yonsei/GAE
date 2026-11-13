"""Read explicit adaptation/evaluation JSON or JSONL cohorts without resplitting."""
import hashlib
import json
from pathlib import Path


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def load_records(path,role,tokenizer=None,dataset='custom',profile='medical',max_length=256):
    if role not in ['adaptation','evaluation']:raise ValueError('Unknown data role')
    path=Path(path);content=path.read_text()
    if path.suffix=='.json':
        payload=json.loads(content)
        if not isinstance(payload,list):raise ValueError('JSON input must be an array of records')
    else:payload=[json.loads(line) for line in content.splitlines() if line.strip()]
    rows=[];groups=set();hashes=set();identifiers=set();contents=set()
    for index,original in enumerate(payload):
        if not isinstance(original,dict):raise ValueError('Input records must be objects')
        source_role=original.get('role',original.get('measurement_role',role))
        if source_role=='final':source_role='evaluation'
        if source_role!=role:raise ValueError('Input role differs from requested cohort')
        row=dict(original);tokens=row.get('tokens')
        if tokens is None:
            text=row.get('text',row.get('prompt'))
            if not isinstance(text,str) or not text.strip() or tokenizer is None:
                raise ValueError('Records require tokens, or text/prompt with a tokenizer')
            bos=tokenizer.bos_token_id
            tokens=tokenizer.encode(text,add_special_tokens=False)
            if bos is not None:tokens=[bos]+tokens
            tokens=tokens[:max_length]
        if not isinstance(tokens,list) or len(tokens)<2 or any(type(t) is not int or t<0 for t in tokens):
            raise ValueError('Tokens must contain at least two nonnegative integer IDs')
        token_hash=row.get('token_hash') or digest(tokens)
        content_hash=digest(tokens)
        text_hash=row.get('text_sha256') or (digest(row.get('text',row.get('prompt'))) if row.get('text',row.get('prompt')) else digest(tokens))
        group=str(row.get('leakage_group',row.get('group_id',text_hash)))
        identity=str(row.get('record_id',row.get('source_id',f'{dataset}:{index}')))
        if identity in identifiers:raise ValueError('Duplicate record ID within cohort')
        if group in groups or token_hash in hashes or content_hash in contents:
            raise ValueError('Duplicate group or token sequence within cohort')
        groups.add(group);hashes.add(token_hash);identifiers.add(identity);contents.add(content_hash)
        row.update(tokens=tokens,record_id=identity,source_id=row.get('source_id',identity),leakage_group=group,
            group_id=group,text_sha256=text_hash,token_hash=token_hash,content_token_sha256=content_hash,
            kind=row.get('kind',dataset),role=role,measurement_role=role)
        rows.append(row)
    if not rows:raise ValueError('Input cohort is empty')
    return rows


def validate_disjoint(adaptation,evaluation):
    for key in ['leakage_group','text_sha256','token_hash']:
        if {r[key] for r in adaptation}&{r[key] for r in evaluation}:raise ValueError('Adaptation/evaluation overlap: '+key)
    def content(row):
        if 'tokens' in row:return digest(row['tokens'])
        if row.get('content_token_sha256'):return row['content_token_sha256']
        raise ValueError('Missing actual token content identity')
    if {content(r) for r in adaptation}&{content(r) for r in evaluation}:
        raise ValueError('Adaptation/evaluation overlap: actual token content')


def validate_frozen_cohort(adaptation_identity,evaluation):
    if not adaptation_identity:raise ValueError('Frozen decoder has no adaptation identity')
    validate_disjoint(adaptation_identity,evaluation)
