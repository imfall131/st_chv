#!/usr/bin/env python
"""딸기 수직농장 데이터 전수 EDA. 원본 TAR를 수정하거나 풀지 않습니다.

실행: python eda_strawberry.py --root . --out eda_results
재집계/보고서: python eda_strawberry.py --root . --out eda_results --reuse
필요 패키지: numpy pandas Pillow reportlab pypdf
Python 3.11 이상 권장. 한국어 글꼴은 --font / --font-bold로 지정할 수 있습니다.
이미지 헤더는 전수 검사, 픽셀 디코딩은 split×품종별 3장(고정 순서) 표본입니다.
통계적 이상치는 오류 확정이 아닌 검토 후보입니다. 원본 삭제·대체는 하지 않습니다.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, math, os, platform
import sys, tarfile, time, zipfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

VERSION = '1.0.0'
ENV_NUM = ['ti_value','hi_value','ci_value','ir_value','tl_value','ei_value','pl_value',
           'cl_value','hl_value','sr_value','pi_value','el_value']
GROW_NUM = ['stem_length','leaf_cnt','leaf_width','leaf_length','stem_thick','fr_weight',
            'fr1_cnt','fr2_cnt','fr3_cnt','leaf_index','spad','petiole_length']
TABLES = ['images','environments','growth_index','annotations','categories','etc_infor','licenses']
NUMERIC = {'images':['width','height','leaf','plant_body','flower','fruit','grpoint','stem'],
           'environments':ENV_NUM,'growth_index':GROW_NUM,
           'annotations':['area','x','y','box_width','box_height','area_ratio']}

def log(*args): print(time.strftime('%H:%M:%S'), *args, flush=True)
def split_of(p): return 'Training' if p.name.startswith(('TL_','TS_')) else 'Validation'
def missing_kind(v):
    if v is None: return 'null'
    if isinstance(v,str) and not v.strip(): return 'blank'
    if isinstance(v,(list,dict)) and not v: return 'empty_container'
    return None
def num(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else np.nan
    except (ValueError,TypeError,OverflowError): return np.nan
def sha(b): return hashlib.sha256(b).hexdigest()
def save_csv(df,p): df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.10g')

class Schema:
    def __init__(self): self.n=collections.Counter(); self.fields={}
    def add(self,table,obj):
        self.n[table]+=1
        for k,v in obj.items():
            c=self.fields.setdefault((table,k),collections.Counter())
            c['present']+=1;c['type_'+type(v).__name__]+=1
            kind=missing_kind(v)
            if kind:c[kind]+=1
    def frame(self):
        rows=[]
        for (t,k),c in sorted(self.fields.items()):
            n=self.n[t]; absent=n-c['present']
            rows.append(dict(table=t,field=k,rows=n,absent=absent,null=c['null'],
                             blank=c['blank'],empty_container=c['empty_container'],
                             missing=absent+c['null']+c['blank']+c['empty_container'],
                             missing_pct=100*(absent+c['null']+c['blank']+c['empty_container'])/n,
                             types='; '.join(f'{k[5:]}:{v}' for k,v in c.items() if k.startswith('type_'))))
        return pd.DataFrame(rows)

def scan(root,out):
    start=time.time(); work=out/'work';work.mkdir(parents=True,exist_ok=True)
    paths=sorted(p for p in root.rglob('*.tar') if out not in p.parents)
    labels=[p for p in paths if p.name.startswith(('TL_','VL_'))]
    sources=[p for p in paths if p.name.startswith(('TS_','VS_'))]
    rows={k:[] for k in TABLES}; schema=Schema(); issues=[]; inventory=[]; containers=[]
    def issue(rule,table,key,detail,archive=''):
        issues.append(dict(rule=rule,table=table,row_key=key,detail=str(detail),archive=archive))
    for ix,p in enumerate(labels):
        split=split_of(p); count=0; errors=0
        try:
            with tarfile.open(p,'r:') as tf:
                for member in tf:
                    if not member.isfile():continue
                    if not member.name.lower().endswith('.json'):
                        issue('unexpected_label_member','archive',member.name,member.size,p.name);continue
                    count+=1; key=split+'/'+p.name+'/'+member.name
                    try:
                        raw=tf.extractfile(member).read(); d=json.loads(raw.decode('utf-8-sig'))
                        if not isinstance(d,dict):raise ValueError('top level is not object')
                    except Exception as e:
                        errors+=1;issue('json_parse_error','label',key,repr(e),p.name);continue
                    im=d.get('images') or {}; envkey='environments' if 'environments' in d else 'envrionments'
                    env=d.get(envkey) or [];ann=d.get('annotations') or [];cats=d.get('categories') or []
                    if not isinstance(im,dict):issue('bad_images_structure','label',key,type(im).__name__,p.name);continue
                    meta=dict(row_key=key,split=split,archive=p.name,member=member.name)
                    schema.add('root',d)
                    containers.append(dict(**meta,environment_key=envkey,env_count=len(env),annotation_count=len(ann),
                                           category_count=len(cats),sha256=sha(raw),root_keys='|'.join(sorted(d))))
                    for t,obj in [('images',im),('growth_index',d.get('growth_index') or {}),('etc_infor',d.get('etc_infor') or {})]:
                        # Keep raw alias fields in schema; normalize only analytical tables.
                        schema.add(t,obj)
                        normalized=dict(obj)
                        if 'crops_id' not in normalized and 'corps_id' in normalized:normalized['crops_id']=normalized['corps_id']
                        r=dict(**meta,**normalized)
                        if t=='images':r.update(json_stem=Path(member.name).stem,env_count=len(env),annotation_count=len(ann))
                        rows[t].append(r)
                    catmap={c.get('id'):c.get('name') for c in cats}
                    if len(catmap)!=len(cats):issue('duplicate_local_category_id','categories',key,'duplicate id',p.name)
                    for t,objs in [('environments',env),('categories',cats),('licenses',d.get('licenses') or [])]:
                        for j,obj in enumerate(objs):
                            schema.add(t,obj);r=dict(**meta,**obj)
                            r['row_key']=key+f'#{t}:{j}';r['label_key']=key
                            r.update(kind_type=im.get('kind_type'),image_farm=im.get('farm_id'),image_id=im.get('image_id'))
                            rows[t].append(r)
                    for j,a in enumerate(ann):
                        schema.add('annotations',a);r=dict(**meta,**a)
                        r['row_key']=key+f'#annotations:{j}';r['label_key']=key
                        r.update(class_name=catmap.get(a.get('category_id')),kind_type=im.get('kind_type'),
                                 farm_id=im.get('farm_id'),img_width=num(im.get('width')),img_height=num(im.get('height')))
                        box=a.get('bbox'); coords=box if isinstance(box,list) and len(box)==4 else [None]*4
                        x,y,w,h=map(num,coords);iw,ih=r['img_width'],r['img_height']; ar=num(a.get('area'))
                        r.update(x=x,y=y,box_width=w,box_height=h,area_ratio=(w*h/(iw*ih) if iw*ih>0 else np.nan))
                        rk=r['row_key']
                        if not all(math.isfinite(z) for z in (x,y,w,h)):issue('bbox_invalid_numeric','annotations',rk,box,p.name)
                        elif w<=0 or h<=0:issue('bbox_nonpositive_size','annotations',rk,box,p.name)
                        elif x<-.02 or y<-.02 or x+w>iw+.02 or y+h>ih+.02:issue('bbox_outside_image','annotations',rk,f'{box}; image={iw}x{ih}',p.name)
                        if math.isfinite(ar) and math.isfinite(w*h) and abs(ar-w*h)>max(1,abs(w*h)*.01):
                            issue('area_bbox_mismatch_gt1pct','annotations',rk,f'area={ar}; bbox_area={w*h}',p.name)
                        if a.get('category_id') not in catmap:issue('unknown_category_id','annotations',rk,a.get('category_id'),p.name)
                        if a.get('image_id')!=im.get('image_id'):issue('annotation_image_id_mismatch','annotations',rk,a.get('image_id'),p.name)
                        seg=a.get('segmentation');flat=seg
                        if isinstance(seg,list) and seg and isinstance(seg[0],list):flat=[v for z in seg for v in z]
                        if not isinstance(flat,list) or len(flat)<6 or len(flat)%2 or any(not math.isfinite(num(z)) for z in flat):
                            issue('segmentation_invalid','annotations',rk,str(seg)[:200],p.name)
                        elif any(num(v)<-.02 or num(v)>(iw if q%2==0 else ih)+.02 for q,v in enumerate(flat)):
                            issue('segmentation_outside_image','annotations',rk,str(seg)[:200],p.name)
                        r.pop('bbox',None);r.pop('segmentation',None);rows['annotations'].append(r)
                    if not ann:issue('empty_annotations','images',key,'0 objects',p.name)
        except Exception as e:issue('tar_read_error','archive',str(p.relative_to(root)),repr(e),p.name)
        inventory.append(dict(path=str(p.relative_to(root)),archive=p.name,split=split,type='label',bytes=p.stat().st_size,members=count,parse_errors=errors))
        if (ix+1)%20==0 or ix+1==len(labels):log('labels',ix+1,'/',len(labels),'records',sum(x['members'] for x in inventory))
    for t,data in rows.items():
        df=pd.DataFrame(data);df.to_pickle(work/(t+'.pkl'));log(t,len(df))
    pd.DataFrame(containers).to_pickle(work/'containers.pkl')
    schema.frame().to_pickle(work/'schema.pkl')
    del rows,containers
    source_rows=[];sample_rows=[];sample_counter=collections.Counter()
    for ix,p in enumerate(sources):
        count=0;split=split_of(p)
        try:
            with tarfile.open(p,'r:') as tf:
                for m in tf:
                    if not m.isfile():continue
                    count+=1;key=split+'/'+p.name+'/'+m.name;status='ok';w=h=None;fmt=None
                    try:
                        with Image.open(tf.extractfile(m)) as pic:w,h=pic.size;fmt=pic.format
                    except Exception as e:status=repr(e);issue('image_header_error','source',key,status,p.name)
                    r=dict(row_key=key,split=split,archive=p.name,member=m.name,stem=Path(m.name).stem,bytes=m.size,
                           width=w,height=h,format=fmt,header_status=status)
                    source_rows.append(r)
                    variety='금실' if '금실' in p.name else '설향' if '설향' in p.name else 'unknown'
                    stratum=(split,variety)
                    if status=='ok' and sample_counter[stratum]<3:
                        sample_counter[stratum]+=1
                        try:
                            b=tf.extractfile(m).read()
                            with Image.open(io.BytesIO(b)) as pic:
                                pic.load(); gray=pic.convert('L');small=gray.resize((256,256));a=np.asarray(small,dtype=float)
                                contrast=float(a.std());brightness=float(a.mean())
                                pic.thumbnail((500,430));thumb=work/f'sample_{len(sample_rows):02d}.jpg';pic.convert('RGB').save(thumb,quality=88)
                            sample_rows.append(dict(**r,kind_type=variety,decode_status='ok',sha256=sha(b),brightness=brightness,contrast=contrast,thumbnail=str(thumb.name)))
                        except Exception as e:
                            issue('sample_pixel_decode_error','source',key,repr(e),p.name)
                            sample_rows.append(dict(**r,kind_type=variety,decode_status=repr(e)))
        except Exception as e:issue('tar_read_error','archive',str(p.relative_to(root)),repr(e),p.name)
        inventory.append(dict(path=str(p.relative_to(root)),archive=p.name,split=split,type='source',bytes=p.stat().st_size,members=count,parse_errors=0))
        if (ix+1)%20==0 or ix+1==len(sources):log('sources',ix+1,'/',len(sources),'images',len(source_rows))
    pd.DataFrame(source_rows).to_pickle(work/'source.pkl')
    pd.DataFrame(sample_rows).to_pickle(work/'samples.pkl')
    pd.DataFrame(issues,columns=['rule','table','row_key','detail','archive']).to_pickle(work/'scan_issues.pkl')
    pd.DataFrame(inventory).to_pickle(work/'inventory.pkl')
    meta=dict(version=VERSION,root=str(root),archive_count=len(paths),archive_bytes=sum(p.stat().st_size for p in paths),
              scan_seconds=time.time()-start,created_seoul=datetime.now(timezone(timedelta(hours=9))).date().isoformat(),python=sys.version,platform=platform.platform())
    (work/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    log('scan completed',round(time.time()-start,1),'seconds')

def analyze(root,out):
    work=out/'work';audit=out/'audit';audit.mkdir(exist_ok=True)
    dfs={t:pd.read_pickle(work/(t+'.pkl')) for t in TABLES}
    im=dfs['images'];en=dfs['environments'];ann=dfs['annotations'];gr=dfs['growth_index']
    source=pd.read_pickle(work/'source.pkl');con=pd.read_pickle(work/'containers.pkl')
    issues=pd.read_pickle(work/'scan_issues.pkl').to_dict('records');schema=pd.read_pickle(work/'schema.pkl')
    inventory=pd.read_pickle(work/'inventory.pkl');meta=json.loads((work/'meta.json').read_text(encoding='utf-8'))
    def add(rule,t,frame,detail=''):
        if not len(frame):return
        for r in frame[['row_key']+(['archive'] if 'archive' in frame else [])].to_dict('records'):
            issues.append(dict(rule=rule,table=t,row_key=r['row_key'],detail=detail,archive=r.get('archive','')))
    # Canonical tables use explicit aliases. Raw per-key presence remains in schema.csv.
    invalid_masks={}
    for t,df in dfs.items():
        if 'corps_id' in df:df.drop(columns='corps_id',inplace=True)
        for c in NUMERIC.get(t,[]):
            if c not in df:df[c]=np.nan
            valid_raw=df[c].notna() & ~df[c].astype(str).str.strip().eq('')
            s=pd.to_numeric(df[c],errors='coerce');bad=valid_raw & (~np.isfinite(s))
            invalid_masks[(t,c)]=bad
            for idx in df.index[bad]:
                issues.append(dict(rule='non_numeric_or_nonfinite:'+c,table=t,row_key=df.at[idx,'row_key'],detail=repr(df.at[idx,c]),archive=df.at[idx,'archive']))
            df[c]=s.where(np.isfinite(s))
    exclude={'row_key','split','archive','member','label_key','json_stem','env_count','annotation_count',
             'kind_type','image_farm','image_id','class_name','farm_id','img_width','img_height','area_ratio',
             'x','y','box_width','box_height'}
    normalized_profile=[];missing_records=0
    with (audit/'missing_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['table','row_key','missing_fields'])
        for t,df in dfs.items():
            rawcols=schema.loc[schema.table.eq(t),'field'].tolist()
            cols=list(dict.fromkeys('crops_id' if c=='corps_id' else c for c in rawcols))
            cols=[c for c in cols if c in df]
            flags=[]
            for c in cols:
                s=df[c];mask=s.isna()
                if s.dtype=='object' or pd.api.types.is_string_dtype(s.dtype):
                    mask=mask | s.map(lambda v:missing_kind(v) is not None)
                bad=invalid_masks.get((t,c),pd.Series(False,index=df.index))
                rawmissing=mask & ~bad
                normalized_profile.append(dict(table=t,field=c,rows=len(df),missing=int(rawmissing.sum()),
                                               missing_pct=100*rawmissing.mean(),invalid_numeric=int(bad.sum()),
                                               unusable=int(mask.sum()),unique_nonmissing=int(s[~mask].astype(str).nunique())))
                flags.append(rawmissing.to_numpy())
            if flags:
                a=np.stack(flags,axis=1)
                rowkeys=df.row_key.to_numpy()
                patterns={}
                for idx in np.flatnonzero(a.any(axis=1)):
                    pattern=tuple(a[idx])
                    if pattern not in patterns:patterns[pattern]='|'.join(c for c,v in zip(cols,pattern) if v)
                    writer.writerow([t,rowkeys[idx],patterns[pattern]])
                    missing_records+=1
    profile=pd.DataFrame(normalized_profile)
    save_csv(schema,audit/'raw_schema.csv');save_csv(profile,audit/'missing_summary.csv')
    # Exact sensor observations: identical farm, timestamp and ALL sensor values,
    # including categorical rainfall/status rp_value (Y/N, never numeric).
    envkeys=['farm_id','receive_date']+ENV_NUM+['rp_value']
    variants=en.drop_duplicates(envkeys).copy()
    variant_rows=variants.duplicated(['farm_id','receive_date'],keep=False)
    unique_en=variants.copy();conflicting_indices=[];remove_indices=[]
    # Missing extra fields do not establish a contradictory sensor observation.
    # Coalesce complementary schemas only when every jointly observed value agrees.
    for _,b in variants[variant_rows].groupby(['farm_id','receive_date']):
        sensorcols=ENV_NUM+['rp_value']
        conflict=any(b[c].dropna().nunique()>1 for c in sensorcols)
        if conflict:conflicting_indices.extend(b.index)
        else:
            first=b.index[0]
            for c in sensorcols:
                values=b[c].dropna()
                if len(values):unique_en.at[first,c]=values.iloc[0]
            remove_indices.extend(b.index[1:])
    unique_en.drop(index=remove_indices,inplace=True)
    unique_en.to_pickle(work/'unique_environment.pkl')
    sensor_conflict=unique_en.index.isin(conflicting_indices)
    add('sensor_timestamp_conflicting_values','environments',unique_en[sensor_conflict], 'Same farm and timestamp has different sensor values')
    save_csv(unique_en[sensor_conflict][['row_key']+envkeys],audit/'sensor_timestamp_conflicts.csv')
    # All numeric profiles and both global / contextual IQR flags, without deleting rows.
    gr['farm_id']=gr.row_key.map(im.set_index('row_key').farm_id)
    stats=[];bounds=[];outlier_counts=[];flagged_rows=0
    with (audit/'outlier_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.writer(f);wr.writerow(['table','row_key','field','value','global_iqr','context_iqr','context','global_lower','global_upper','context_lower','context_upper'])
        for t,cols in NUMERIC.items():
            df=dfs[t];baseline=unique_en if t=='environments' else df
            groupcol={'images':'kind_type','environments':'farm_id','growth_index':'farm_id','annotations':'class_name'}[t]
            for c in cols:
                s=baseline[c].dropna(); x=df[c];n=len(s)
                if n:
                    q1,med,q3=s.quantile([.25,.5,.75]);iqr=q3-q1;lo=q1-1.5*iqr;hi=q3+1.5*iqr
                    # Zero-IQR fields are not forced into statistical outlier detection.
                    gm=(x.lt(lo)|x.gt(hi)) if n>=20 and iqr>0 else pd.Series(False,index=df.index)
                    stats.append(dict(table=t,field=c,n=n,missing_baseline=len(baseline)-n,mean=s.mean(),std=s.std(),min=s.min(),q1=q1,median=med,q3=q3,max=s.max(),zero=int(s.eq(0).sum()),unique=s.nunique(),global_outlier_raw=int(gm.sum()),iqr=iqr))
                else:
                    lo=hi=np.nan;gm=pd.Series(False,index=df.index)
                    stats.append(dict(table=t,field=c,n=0,missing_baseline=len(baseline),mean=np.nan,std=np.nan,min=np.nan,q1=np.nan,median=np.nan,q3=np.nan,max=np.nan,zero=0,unique=0,global_outlier_raw=0,iqr=np.nan))
                bounds.append(dict(table=t,field=c,context='ALL',n=n,lower=lo,upper=hi,enabled=bool(n>=20 and n and iqr>0)))
                cm=pd.Series(False,index=df.index);ctxlo=pd.Series(np.nan,index=df.index);ctxhi=ctxlo.copy()
                if groupcol:
                    for group,b in baseline.groupby(groupcol,dropna=False):
                        bs=b[c].dropna();nn=len(bs)
                        if nn:
                            lq,uq=bs.quantile([.25,.75]);spread=uq-lq;l=lq-1.5*spread;h=uq+1.5*spread
                        else:l=h=np.nan;spread=0
                        enabled=nn>=20 and spread>0
                        bounds.append(dict(table=t,field=c,context=str(group),n=nn,lower=l,upper=h,enabled=bool(enabled)))
                        sel=df[groupcol].eq(group) if pd.notna(group) else df[groupcol].isna()
                        ctxlo.loc[sel]=l;ctxhi.loc[sel]=h
                        if enabled:cm.loc[sel]=df.loc[sel,c].lt(l)|df.loc[sel,c].gt(h)
                union=gm|cm
                outlier_counts.append(dict(table=t,field=c,raw_nonmissing=int(x.notna().sum()),global_iqr=int(gm.sum()),context_iqr=int(cm.sum()),union=int(union.sum())))
                for idx in df.index[union]:
                    wr.writerow([t,df.at[idx,'row_key'],c,x.at[idx],int(gm.at[idx]),int(cm.at[idx]),df.at[idx,groupcol] if groupcol else '',lo,hi,ctxlo.at[idx],ctxhi.at[idx]])
                    flagged_rows+=1
                if t=='environments':
                    bad=(x.lt(0)|x.gt(100)) if c=='hi_value' else x.lt(0) if c in ['ci_value','ir_value','ei_value','pl_value'] else pd.Series(False,index=df.index)
                    add('domain_range:'+c,t,df[bad], 'Humidity outside [0,100] or negative nonnegative sensor')
                if t=='growth_index':
                    add('negative_growth:'+c,t,df[x.lt(0)], 'Negative growth measurement')
                    if c in ['leaf_cnt','fr1_cnt','fr2_cnt','fr3_cnt']:add('non_integer_count:'+c,t,df[x.notna() & x.mod(1).ne(0)],'Count is not integer')
                if t=='images' and c in ['width','height']:add('nonpositive_dimension:'+c,t,df[x.le(0)], 'Image dimension <=0')
    stats=pd.DataFrame(stats);bounds=pd.DataFrame(bounds);outlier_counts=pd.DataFrame(outlier_counts)
    save_csv(stats,audit/'numeric_summary.csv');save_csv(bounds,audit/'outlier_bounds.csv');save_csv(outlier_counts,audit/'outlier_summary.csv')
    # Datetimes, sensor alignment, and growth timing.
    cap=pd.to_datetime(im.date_captured,errors='coerce');recv=pd.to_datetime(en.receive_date,errors='coerce')
    add('invalid_capture_date','images',im[cap.isna()],'date_captured cannot be parsed')
    add('invalid_receive_date','environments',en[recv.isna()],'receive_date cannot be parsed')
    capturemap=pd.Series(cap.to_numpy(),index=im.row_key)
    en['capture_time']=en.label_key.map(capturemap);en['receive_time']=recv
    en['lag_minutes']=(en.capture_time-recv).dt.total_seconds()/60
    en['abs_lag']=en.lag_minutes.abs()
    valid=en.dropna(subset=['abs_lag']);near=en.loc[valid.groupby('label_key',sort=False).abs_lag.idxmin()].copy()
    add('nearest_sensor_after_capture','images',near[near.lag_minutes.lt(0)].rename(columns={'row_key':'sensor_key','label_key':'row_key'}),'Nearest sensor is after image capture')
    add('nearest_sensor_abs_lag_gt15min','images',near[near.abs_lag.gt(15)].rename(columns={'row_key':'sensor_key','label_key':'row_key'}),'Heuristic alignment review threshold 15 minutes')
    add('no_environment_records','images',im[im.env_count.eq(0)],'No sensor rows linked')
    add('environment_farm_mismatch','environments',en[en.farm_id.ne(en.image_farm)],'Image and sensor farm_id differ')
    add('missing_class_name','annotations',ann[ann.class_name.isna() | ann.class_name.eq('')],'Category name is null or blank')
    add('unexpected_rp_value','environments',en[en.rp_value.notna() & ~en.rp_value.isin(['Y','N'])],'Expected observed categorical values Y/N')
    growth_time={};growth_dates={}
    for field in ['measured_date','bloom1_date','bloom2_date','bloom3_date']:
        raw=gr[field] if field in gr else pd.Series(index=gr.index,dtype='object')
        parsed=pd.to_datetime(raw,errors='coerce',format='mixed')
        growth_dates[field]=parsed
        bad=raw.notna() & parsed.isna();add('invalid_growth_date:'+field,'growth_index',gr[bad],'Nonmissing date cannot be parsed')
        capture_day=gr.row_key.map(capturemap).dt.floor('D')
        future=parsed.dt.floor('D').gt(capture_day)
        growth_time[field]=dict(nonmissing=int(raw.notna().sum()),invalid=int(bad.sum()),min=str(parsed.min()),max=str(parsed.max()),after_capture=int(future.sum()))
        for idx in gr.index[future]:
            issues.append(dict(rule='growth_date_after_capture:'+field,table='growth_index',row_key=gr.at[idx,'row_key'],detail=f'{field}={raw.at[idx]}; capture={capture_day.at[idx]}',archive=gr.at[idx,'archive']))
        if field=='measured_date':
            delta=(parsed-gr.row_key.map(capturemap)).dt.total_seconds()/86400
            growth_time['measurement_vs_capture']=dict(n=int(delta.notna().sum()),median_days=float(delta.median()),future=int(delta.gt(0).sum()),abs_gt7days=int(delta.abs().gt(7).sum()))
    for earlier,later in [('bloom1_date','bloom2_date'),('bloom2_date','bloom3_date')]:
        wrong=growth_dates[later].lt(growth_dates[earlier]);growth_time[later+'_before_'+earlier]=int(wrong.sum())
        for idx in gr.index[wrong]:
            issues.append(dict(rule='bloom_order:'+later+'<'+earlier,table='growth_index',row_key=gr.at[idx,'row_key'],detail=f'{earlier}={gr.at[idx,earlier]}; {later}={gr.at[idx,later]}',archive=gr.at[idx,'archive']))
    save_csv(near[['label_key','split','archive','farm_id','capture_time','receive_time','lag_minutes']],audit/'nearest_sensor_alignment.csv')
    # Duplicates / split leakage: identity overlap and related series are distinct.
    duplicates=[];leakage=[]
    im['series_id']=im.json_stem.str.rsplit('_',n=1).str[0]
    duplicate_rules={'image_id':['image_id'],'json_stem':['json_stem'],'fname':['fname'],'capture_series':['series_id','date_captured']}
    for name,keys in duplicate_rules.items():
        du=im.duplicated(keys,keep=False);add('duplicate_image_key:'+name,'images',im[du], '|'.join(keys))
        duplicates.append(dict(key=name,duplicate_rows=int(du.sum()),duplicate_groups=int(im.loc[du].groupby(keys).ngroups)))
        tr=im[im.split.eq('Training')];va=im[im.split.eq('Validation')]
        trkeys=set(map(tuple,tr[keys].to_numpy()));vakeys=set(map(tuple,va[keys].to_numpy()));shared=trkeys&vakeys
        mask=im[keys].apply(tuple,axis=1).isin(shared)
        leakage.append(dict(key=name,shared_keys=len(shared),affected_rows=int(mask.sum())))
        add('cross_split_same_image_key:'+name,'images',im[mask], '|'.join(keys))
    for c in ['farm_id','crops_id','series_id']:
        shared=set(im.loc[im.split.eq('Training'),c].dropna())&set(im.loc[im.split.eq('Validation'),c].dropna())
        leakage.append(dict(key=c,shared_keys=len(shared),affected_rows=int(im[c].isin(shared).sum())))
    trsen=set(map(tuple,en.loc[en.split.eq('Training'),['farm_id','receive_date']].to_numpy()))
    vasen=set(map(tuple,en.loc[en.split.eq('Validation'),['farm_id','receive_date']].to_numpy()))
    leakage.append(dict(key='sensor_farm_timestamp',shared_keys=len(trsen&vasen),affected_rows=int(en[['farm_id','receive_date']].apply(tuple,axis=1).isin(trsen&vasen).sum())))
    hashdu=con.duplicated('sha256',keep=False);add('duplicate_json_sha256','label',con[hashdu],'Byte-identical JSON')
    # Header-level source matching uses JSON stem, independently checks fname discrepancy.
    source_du=source.duplicated(['split','stem'],keep=False)
    add('duplicate_source_stem','source',source[source_du],'Duplicate source key inside split')
    pairs=im.merge(source,how='outer',left_on=['split','json_stem'],right_on=['split','stem'],indicator=True,suffixes=('_label','_source'))
    for side,rule in [('left_only','label_without_source'),('right_only','source_without_label')]:
        sub=pairs[pairs['_merge'].eq(side)]
        for r in sub.to_dict('records'):
            issues.append(dict(rule=rule,table='pair',row_key=r.get('row_key_label') if side=='left_only' else r.get('row_key_source'),detail=str(r.get('json_stem') if side=='left_only' else r.get('stem')),archive=str(r.get('archive_label') if side=='left_only' else r.get('archive_source'))))
    both=pairs[pairs['_merge'].eq('both')];dim_bad=both[(both.width_label.ne(both.width_source))|(both.height_label.ne(both.height_source))]
    for r in dim_bad.to_dict('records'):
        issues.append(dict(rule='label_source_dimension_mismatch',table='pair',row_key=r['row_key_label'],detail=f"label={r['width_label']}x{r['height_label']}; source={r['width_source']}x{r['height_source']}",archive=r['archive_label']))
    fname_diff=im.fname.str.replace(r'\.[^.]+$','',regex=True).ne(im.json_stem)
    add('fname_differs_json_stem','images',im[fname_diff],'Stored fname uses another naming convention; canonical matching uses JSON stem')
    source_names=set(zip(source.split,source.member.map(lambda s:Path(s).name)))
    fname_match=sum((s,f) in source_names for s,f in zip(im.split,im.fname))
    save_csv(pairs[pairs['_merge'].ne('both')],audit/'unmatched_sources_labels.csv')
    save_csv(dim_bad,audit/'dimension_mismatches.csv')
    # Cadence and per-farm numerical correlations use de-duplicated sensor observations.
    unique_en['receive_time']=pd.to_datetime(unique_en.receive_date,errors='coerce')
    cadence=[]
    for farm,b in unique_en.groupby('farm_id'):
        ts=b.receive_time.dropna().drop_duplicates().sort_values();delta=ts.diff().dt.total_seconds().div(60).dropna()
        cadence.append(dict(farm_id=farm,rows=len(b),unique_times=len(ts),start=str(ts.min()),end=str(ts.max()),
                            median_gap_minutes=delta.median(),p95_gap_minutes=delta.quantile(.95),max_gap_minutes=delta.max(),gaps_gt60min=int(delta.gt(60).sum())))
    cadence=pd.DataFrame(cadence);save_csv(cadence,audit/'sensor_cadence.csv')
    corr=unique_en[ENV_NUM].corr(method='pearson',min_periods=30);corr.to_csv(audit/'sensor_correlation.csv',encoding='utf-8-sig')
    category_map=dfs['categories'].groupby(['id','name']).size().reset_index(name='records')
    save_csv(category_map,audit/'local_category_mapping.csv')
    issueframe=pd.DataFrame(issues,columns=['rule','table','row_key','detail','archive'])
    save_csv(issueframe,audit/'quality_issues.csv');save_csv(inventory,audit/'archive_inventory.csv')
    issue_counts=issueframe.groupby('rule').agg(records=('row_key','size'),unique_rows=('row_key','nunique')).reset_index().sort_values('records',ascending=False)
    save_csv(issue_counts,audit/'quality_summary.csv')
    # Exact per-image distribution and all sensor values support independent reanalysis.
    im['capture_time']=cap;im['month']=cap.dt.strftime('%Y-%m');im['hour']=cap.dt.hour
    save_csv(im,audit/'image_metadata.csv')
    save_csv(unique_en.drop(columns=['receive_time']),audit/'unique_environment_observations.csv')
    save_csv(pd.read_pickle(work/'samples.pkl'),audit/'image_decode_samples.csv')
    group=im.groupby(['split','kind_type','farm_id','growth_stage'],dropna=False).agg(images=('row_key','size'),series=('series_id','nunique'),first_capture=('capture_time','min'),last_capture=('capture_time','max')).reset_index()
    save_csv(group,audit/'image_groups.csv')
    # Reconciliation checks are intentionally independent of the report formatting.
    assert len(im)==inventory.loc[inventory.type.eq('label'),'members'].sum(), 'Parsed label count mismatch'
    assert len(source)==inventory.loc[inventory.type.eq('source'),'members'].sum(), 'Source member count mismatch'
    assert len(ann)==len(dfs['categories']), 'Annotation/category record count mismatch'
    assert im.env_count.sum()==len(en), 'Environment row reconciliation failed'
    assert im.annotation_count.sum()==len(ann), 'Annotation row reconciliation failed'
    assert profile.missing.ge(0).all() and profile.missing.le(profile.rows).all()
    for r in profile.itertuples():
        if r.field=='crops_id' and r.table in ['growth_index','etc_infor']:continue
        original=schema.loc[schema.table.eq(r.table) & schema.field.eq(r.field),'missing']
        if len(original):assert r.missing==int(original.iloc[0]), f'Raw / normalized missing mismatch: {r.table}.{r.field}'
    summary=dict(meta=meta,counts={t:len(df) for t,df in dfs.items()},source_images=len(source),
                 unique_env=len(unique_en),sensor_repeated_rows=len(en)-len(unique_en),
                 sensor_schema_variant_rows=int(variant_rows.sum()),sensor_conflicting_rows=len(conflicting_indices),
                 missing_records=missing_records,statistical_outlier_cells=flagged_rows,
                 date_start=str(cap.min()),date_end=str(cap.max()),invalid_dates=int(cap.isna().sum()),
                 pairing={k:int(v) for k,v in pairs['_merge'].value_counts().items()},dimension_mismatches=len(dim_bad),
                 fname_differs_json_stem=int(fname_diff.sum()),fname_direct_match=int(fname_match),
                 duplicates=duplicates,leakage=leakage,json_sha_duplicate_rows=int(hashdu.sum()),
                 lag=dict(n=len(near),median=float(near.lag_minutes.median()),p95=float(near.abs_lag.quantile(.95)),
                          max_abs=float(near.abs_lag.max()),after_capture=int(near.lag_minutes.lt(0).sum()),
                          gt15=int(near.abs_lag.gt(15).sum()),gt60=int(near.abs_lag.gt(60).sum()),gt240=int(near.abs_lag.gt(240).sum())),
                 growth_time=growth_time,top_keys=con.root_keys.value_counts().to_dict(),environment_keys=con.environment_key.value_counts().to_dict())
    (work/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    # Cache report-ready normalized tables. They are internal, reproducible intermediates.
    for name,df in [('report_images',im),('report_environments',unique_en),('report_annotations',ann),('report_growth',gr),('report_profile',profile),('report_stats',stats),('report_issues',issue_counts),('report_outliers',outlier_counts),('report_groups',group),('report_cadence',cadence),('report_near',near),('report_correlation',corr)]:
        df.to_pickle(work/(name+'.pkl'))
    log('analysis completed','outlier cells',flagged_rows,'quality records',len(issueframe))

def build_report(root,out,args):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image as RLImage,KeepTogether
    from reportlab.graphics.shapes import Drawing,Rect,String,Line,Circle,PolyLine
    from pypdf import PdfReader,PdfWriter
    from xml.sax.saxutils import escape
    import datetime
    pdfmetrics.registerFont(TTFont('MG',args.font));pdfmetrics.registerFont(TTFont('MGB',args.font_bold))
    work=out/'work';audit=out/'audit';s=json.loads((work/'summary.json').read_text(encoding='utf-8'))
    get=lambda n:pd.read_pickle(work/('report_'+n+'.pkl'))
    im=get('images');en=get('environments');ann=get('annotations');gr=get('growth')
    profile=get('profile');stats=get('stats');issues=get('issues');outs=get('outliers');near=get('near')
    inventory=pd.read_pickle(work/'inventory.pkl');schema=pd.read_pickle(work/'schema.pkl');samples=pd.read_pickle(work/'samples.pkl')
    analysis_date=s['meta']['created_seoul'];analysis_day=datetime.date.fromisoformat(analysis_date)
    quality=pd.read_csv(audit/'quality_issues.csv',keep_default_na=False)
    count=lambda rule:int(issues.loc[issues.rule.eq(rule),'records'].sum())
    N=len(im);A=len(ann);W=507
    navy=colors.HexColor('#18374D');blue=colors.HexColor('#327DA8');teal=colors.HexColor('#319C91')
    orange=colors.HexColor('#C87C35');red=colors.HexColor('#B64A4A');gray=colors.HexColor('#6B7280')
    pale=colors.HexColor('#F1F5F8');grid=colors.HexColor('#D9D9D9')
    palette=[blue,teal,orange,red,colors.HexColor('#8064A2'),colors.HexColor('#91A550'),colors.HexColor('#A6678E'),colors.HexColor('#5B8796'),colors.HexColor('#997B59')]
    styles={
      'title':ParagraphStyle('title',fontName='MGB',fontSize=25,leading=34,textColor=colors.black,spaceAfter=16,wordWrap='CJK'),
      'h1':ParagraphStyle('h1',fontName='MGB',fontSize=17,leading=24,textColor=colors.black,spaceAfter=12,wordWrap='CJK',keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='MGB',fontSize=11,leading=16,textColor=colors.black,spaceBefore=10,spaceAfter=6,wordWrap='CJK',keepWithNext=True),
      'body':ParagraphStyle('body',fontName='MG',fontSize=9,leading=14.5,spaceAfter=8,wordWrap='CJK'),
      'small':ParagraphStyle('small',fontName='MG',fontSize=7.7,leading=11.5,spaceAfter=6,wordWrap='CJK',textColor=gray),
      'cell':ParagraphStyle('cell',fontName='MG',fontSize=7.4,leading=10.3,wordWrap='CJK'),
      'number':ParagraphStyle('number',fontName='MG',fontSize=7.4,leading=10.3,wordWrap='CJK',alignment=2),
      'head':ParagraphStyle('head',fontName='MGB',fontSize=7.5,leading=10.5,wordWrap='CJK',textColor=colors.white),
    }
    story=[];section_titles=[]
    def fmt(v,d=2):
        if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)):return '-'
        if isinstance(v,(int,np.integer)):return f'{v:,}'
        if isinstance(v,(float,np.floating)):
            if abs(v)>=1e6:return f'{v:,.0f}'
            return f'{v:,.{d}f}'.rstrip('0').rstrip('.')
        return str(v)
    def P(txt,style='body'):return Paragraph(escape(str(txt)).replace('\n','<br/>'),styles[style])
    def para(txt,style='body'):story.append(P(txt,style))
    def title(txt):
        if story:story.append(PageBreak())
        section_titles.append(txt);para(txt,'h1')
    def table(headers,data,widths=None,font=7.4,compact=False):
        widths=widths or [W/len(headers)]*len(headers)
        cells=[[P(h,'head') for h in headers]]+[[P(fmt(v),'number' if isinstance(v,(int,float,np.number)) or str(v).endswith('%') else 'cell') for v in row] for row in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('GRID',(0,0),(-1,-1),.35,grid),
                               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                               ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                               ('TOPPADDING',(0,0),(-1,-1),4 if compact else 5),('BOTTOMPADDING',(0,0),(-1,-1),4 if compact else 5)]))
        story.extend([t,Spacer(1,9)]);return t
    def hbar(labels,values,height=200,color=blue,unit='건',maxv=None):
        d=Drawing(W,height);left=100;right=62;top=17;bottom=12;plotw=W-left-right
        mx=maxv or max(values or [1]) or 1;rh=(height-top-bottom)/max(len(labels),1)
        for i,(label,val) in enumerate(zip(labels,values)):
            y=height-top-(i+1)*rh
            d.add(String(left-8,y+rh*.35,str(label),fontName='MG',fontSize=8,textAnchor='end',fillColor=navy))
            d.add(Rect(left,y+rh*.18,plotw*float(val)/mx,rh*.60,fillColor=color,strokeColor=None))
            d.add(String(left+plotw*float(val)/mx+6,y+rh*.35,fmt(val)+unit,fontName='MG',fontSize=7.7,fillColor=navy))
        return d
    def histpanels(items,height=420):
        d=Drawing(W,height);cols=2;rows=math.ceil(len(items)/cols);cw=W/2;ch=height/max(rows,1)
        for k,(label,values) in enumerate(items):
            vals=pd.Series(values).dropna().to_numpy(dtype=float);col=k%2;row=k//2
            ox=col*cw+34;oy=height-(row+1)*ch+32;pw=cw-50;ph=ch-65
            d.add(String(col*cw+8,oy+ph+18,label,fontName='MGB',fontSize=8.2,fillColor=navy))
            if not len(vals):continue
            l,h=np.quantile(vals,[.01,.99])
            if l==h:l,h=float(vals.min())-.5,float(vals.max())+.5
            counts,edges=np.histogram(vals,bins=24,range=(l,h));mx=max(counts.max(),1)
            for j,n in enumerate(counts):d.add(Rect(ox+j*pw/24,oy,pw/24-.3,ph*n/mx,fillColor=blue,strokeColor=None))
            d.add(Line(ox,oy,ox+pw,oy,strokeColor=grid))
            for x,anchor,pos in [(l,'start',ox),(h,'end',ox+pw)]:d.add(String(pos,oy-11,fmt(x),fontName='MG',fontSize=7,textAnchor=anchor,fillColor=gray))
            omitted=int(((vals<l)|(vals>h)).sum())
            d.add(String(ox,oy-23,f'n={len(vals):,}  표시범위 밖 {omitted:,}',fontName='MG',fontSize=6.8,fillColor=gray))
        return d
    def linechart(df,xcol,ycols,height=200):
        d=Drawing(W,height);left=34;bottom=35;top=25;pw=W-left-15;ph=height-bottom-top
        x=pd.to_datetime(df[xcol]);lo=x.min();hi=x.max();span=max((hi-lo).total_seconds(),1)
        vals=df[ycols].to_numpy(dtype=float);mn=float(np.nanmin(vals));mx=float(np.nanmax(vals));rng=max(mx-mn,1)
        d.add(Line(left,bottom,left+pw,bottom,strokeColor=grid))
        for v in [mn,(mn+mx)/2,mx]:
            y=bottom+ph*(v-mn)/rng;d.add(Line(left,y,left+pw,y,strokeColor=grid,strokeWidth=.3))
            d.add(String(left-5,y-2,fmt(v),fontName='MG',fontSize=7,textAnchor='end',fillColor=gray))
        for j,c in enumerate(ycols):
            points=[]
            for dt,v in zip(x,df[c]):
                if pd.notna(v):points.extend([left+pw*(dt-lo).total_seconds()/span,bottom+ph*(float(v)-mn)/rng])
            if len(points)>=4:d.add(PolyLine(points,strokeColor=palette[j%len(palette)],strokeWidth=1,fillColor=None))
        for j,c in enumerate(ycols):
            xx=left+(j%5)*94;yy=height-8-(j//5)*11
            d.add(Line(xx,yy,xx+12,yy,strokeColor=palette[j%len(palette)],strokeWidth=2))
            d.add(String(xx+16,yy-2,str(c),fontName='MG',fontSize=6.8,fillColor=gray))
        for dt,anchor,pos in [(lo,'start',left),(hi,'end',left+pw)]:d.add(String(pos,bottom-13,dt.strftime('%Y-%m-%d'),fontName='MG',fontSize=7,textAnchor=anchor,fillColor=gray))
        return d
    def heatmap_missing():
        farms=sorted(en.farm_id.dropna().unique());cols=ENV_NUM+['rp_value'];d=Drawing(W,235)
        left=55;bottom=50;cw=(W-left-8)/len(cols);rh=18
        for j,c in enumerate(cols):
            d.add(String(left+j*cw+cw/2,bottom-12,c.replace('_value',''),fontName='MG',fontSize=7,textAnchor='middle',fillColor=navy))
        for i,farm in enumerate(farms):
            y=bottom+(len(farms)-i-1)*rh;b=en[en.farm_id.eq(farm)]
            d.add(String(left-5,y+6,farm,fontName='MG',fontSize=7,textAnchor='end',fillColor=navy))
            for j,c in enumerate(cols):
                ratio=b[c].isna().mean();fill=colors.Color(.93-.65*ratio,.97-.45*ratio,.98-.30*ratio)
                d.add(Rect(left+j*cw,y,cw,rh,fillColor=fill,strokeColor=colors.white,strokeWidth=.6))
                d.add(String(left+j*cw+cw/2,y+6,str(round(ratio*100)),fontName='MG',fontSize=6.7,textAnchor='middle',fillColor=colors.white if ratio>.65 else navy))
        d.add(String(left,15,'셀 숫자: 결측률 %   분모: 농장별 중복 제거 센서 관측',fontName='MG',fontSize=7,fillColor=gray))
        return d
    # Opening page
    para('딸기 수직농장 데이터 EDA 결과 보고서','title')
    para('전체 파일 구조와 분포 결측치 이상치 데이터 연결 품질 분석','h2')
    para(f'분석일 {analysis_day.year}년 {analysis_day.month}월 {analysis_day.day}일 | 대상 099 지능형 수직농장 통합 데이터 딸기','small')
    para(f'이미지 {N:,}장과 객체 주석 {A:,}건을 전수 집계했다. 이미지와 라벨은 파일명 기준으로 모두 연결되지만, 일부 좌표와 이미지 크기 정보에 오류가 있고 생육 측정값의 결측이 크다. 모델 학습 전 좌표 보정, 생육값 정리, 학습 검증 분리 기준 점검을 먼저 진행하는 것이 적절하다.')
    table(['전수 분석 대상','규모'],[
      ['TAR 파일',f"{s['meta']['archive_count']:,}개 / {s['meta']['archive_bytes']/1e9:,.2f} GB / {s['meta']['archive_bytes']/1024**3:,.2f} GiB"],
      ['이미지 및 JSON 라벨',f'{N:,}쌍'],['환경 센서 원시 연결 행',f"{s['counts']['environments']:,}행"],
      ['중복 제거 센서 관측',f"{s['unique_env']:,}건"],['객체 주석',f'{A:,}건'],
      ['촬영 기간',f"{s['date_start']} 부터 {s['date_end']}"],
      ['분포',f"품종 {im.kind_type.nunique()}종 / 농장 {im.farm_id.nunique()}곳 / 촬영 계열 {im.series_id.nunique()}개 / 생육 단계 {im.growth_stage.nunique()}종"]],[140,367])
    para('우선 확인할 결과','h2')
    fr_missing=float(profile.loc[profile.table.eq('growth_index') & profile.field.eq('fr_weight'),'missing_pct'].iloc[0])
    findings=[
      f"생육 주요 지표의 결측률은 약 86% 이상이고 fr_weight는 {fr_missing:.2f}% 결측이다. 단순 완전 사례 삭제는 관측 대부분을 잃게 한다.",
      f"라벨과 실제 이미지 크기 불일치 {s['dimension_mismatches']:,}건, 음수 또는 0 이하 박스 크기 {count('bbox_nonpositive_size'):,}건, 이미지 밖 박스 {count('bbox_outside_image'):,}건을 확인했다.",
      f"숫자로 해석되지 않는 생육값 {int(issues.loc[issues.rule.str.startswith('non_numeric_or_nonfinite:'),'records'].sum()):,}건을 원문과 함께 기록했다. 단위가 붙은 과중과 백틱 입력이 포함된다.",
      f"가장 가까운 센서값과 촬영 시각의 차이가 15분을 넘는 이미지가 {s['lag']['gt15']:,}장이다. 가장 큰 차이는 {s['lag']['max_abs']/60:.2f}시간이다.",
      f"서로 다른 split에 같은 촬영 계열과 시각이 겹치는 키가 {next(x['shared_keys'] for x in s['leakage'] if x['key']=='capture_series'):,}개다. 동일 이미지 ID 중복과는 구분해 평가 누수 가능성을 검토해야 한다."]
    for i,text in enumerate(findings,1):para(f'{i}. {text}')
    para('본문의 모든 표와 그래프는 전체 집계를 바탕으로 작성했다. 전체 행 단위 결측 목록과 이상치 값은 문서에 첨부된 eda_audit.zip에 수록했다. PDF 뷰어의 첨부 파일 패널에서 열 수 있으며, 같은 ZIP을 결과 폴더에도 제공한다.','small')
    # Scope
    title('1 데이터 구조와 분석 기준')
    inv=inventory.groupby(['split','type']).agg(archives=('archive','size'),members=('members','sum'),bytes=('bytes','sum')).reset_index()
    table(['분할','종류','TAR 개수','파일 수','용량 GB'],[[r.split,r.type,r.archives,r.members,f'{r.bytes/1e9:,.3f}'] for r in inv.itertuples()],[90,80,75,120,142])
    table(['논리 테이블','행 수','주요 내용'],[[t,s['counts'][t],desc] for t,desc in [
      ('images','이미지 ID 품종 농장 촬영 시각 해상도 생육 단계'),('environments','이미지별 2개 센서 관측과 센서값'),
      ('growth_index','이미지별 생육값과 측정일'),('annotations','박스 면적 좌표 분할 좌표와 로컬 category_id'),
      ('categories','각 JSON 내부 category_id와 클래스 이름 대응'),('etc_infor','부가 정보'),('licenses','라이선스 메타데이터')]],[100,90,317])
    para('정규화와 집계 단위','h2')
    para(f"environments와 envrionments를 같은 센서 테이블로 통일했고, crops_id와 corps_id를 같은 식별자 필드로 통일했다. 원문 키별 존재 여부는 부록에 따로 표시한다. envrionments를 사용하는 JSON은 {s['environment_keys'].get('envrionments',0):,}개다.")
    para('결측은 키 부재, null, 빈 문자열, 빈 컨테이너를 구분했다. 숫자가 아닌 문자열은 원시 결측에 포함하지 않고 형식 오류로 기록했다. 숫자 0은 결측으로 바꾸지 않았다. 원본 데이터는 수정하지 않았다.')
    para('센서는 농장과 수신 시각 및 센서값으로 중복을 확인했다. 같은 시각에서 관측된 값이 모두 일치하고 일부 추가 필드만 빠진 경우에는 비어 있지 않은 값으로 결합했다. 날짜가 다르거나 공동 관측값이 다른 경우는 합치지 않는다. 센서 기초통계와 상관은 중복 제거 관측 기준이다.')
    para('필드 단위와 센서 코드의 공식 명세가 폴더에 포함되지 않아 센서는 원래 필드명으로 표기한다. 생육값도 단위 변환 없이 원문 숫자를 사용한다. hi_value의 [0,100] 점검은 상대습도라는 가정에 따른 검토 규칙이며, 다른 센서에 임의의 적정 생육 범위를 적용하지 않았다.','small')
    para('검사 범위','h2')
    para(f"JSON 파싱, 모든 원본 이미지 헤더, 라벨 연결, 메타데이터 및 모든 수치 필드를 검사했다. 전체 {s['meta']['archive_bytes']/1e9:,.1f}GB 이미지의 픽셀 디코딩 및 파일 해시 전수 검사는 수행하지 않았다. 픽셀 디코딩과 밝기 검사는 split과 품종별 고정 순서 3장씩 총 {len(samples)}장에 한정한다. 따라서 전수 이미지 무결성이나 시각적 중복 부재를 보증하는 결과는 아니다.")
    # Distribution
    title('2 품종 농장 생육 단계 분포')
    splitgroup=im.groupby(['split','kind_type']).size().unstack(fill_value=0)
    table(['분할','금실','설향','합계','전체 비율'],[[idx,int(r.get('금실',0)),int(r.get('설향',0)),int(r.sum()),f'{100*r.sum()/N:.2f}%'] for idx,r in splitgroup.iterrows()],[95,90,100,105,117])
    farms=im.groupby('farm_id').size().sort_values(ascending=False)
    story.append(hbar(farms.index.tolist(),farms.tolist(),215))
    stages=im.groupby(['growth_stage','split']).size().unstack(fill_value=0)
    table(['생육 단계','Training','Validation','합계'],[[idx,int(r.get('Training',0)),int(r.get('Validation',0)),int(r.sum())] for idx,r in stages.iterrows()],[150,119,119,119])
    para(f"가장 큰 농장 {farms.index[0]}가 전체의 {farms.iloc[0]/N*100:.2f}%를 차지한다. 금실 이미지는 {im.loc[im.kind_type.eq('금실'),'farm_id'].nunique()}개 농장에서만 관측되어 품종 효과와 농장 효과를 분리하기 어렵다. 단순 전체 정확도 외에 농장 품종 생육 단계별 성능도 함께 평가해야 한다.")
    # Time
    title('3 촬영 시점과 분할 구성')
    monthly=im.groupby(['month','split']).size().unstack(fill_value=0)
    table(['촬영 월','Training','Validation','합계'],[[idx,int(r.get('Training',0)),int(r.get('Validation',0)),int(r.sum())] for idx,r in monthly.iterrows()],[140,120,120,127])
    story.append(hbar(monthly.index.tolist(),monthly.sum(axis=1).tolist(),150))
    hourly=im.groupby('hour').size().sort_index()
    story.append(hbar([f'{int(h):02d}시' for h in hourly.index],hourly.tolist(),310))
    para('시간대는 원문 날짜 문자열을 그대로 해석했다. 원천 타임존 메타데이터는 없으므로 센서와 이미지의 시각 차이도 같은 기준 시계라는 가정으로 계산했다. 수집은 2021년 하반기부터 2022년 1월까지로 제한되어 다른 계절 성능은 이 자료만으로 판단할 수 없다.','small')
    # Image
    title('4 이미지 해상도와 표본 확인')
    si=stats[stats.table.eq('images') & stats.field.isin(['width','height'])]
    table(['항목','최소','Q1','중앙값','Q3','최대'],[[r.field,r.min,r.q1,r.median,r.q3,r.max] for r in si.itertuples()],[90,80,80,90,80,87])
    story.append(histpanels([('width 픽셀',im.width),('height 픽셀',im.height)],180))
    para(f"헤더 읽기 오류는 {count('image_header_error'):,}건이다. JSON 해상도와 실제 JPEG 헤더 해상도의 불일치는 {s['dimension_mismatches']:,}장으로 전체 {s['dimension_mismatches']/N*100:.3f}%다. 박스의 유효성은 라벨 해상도 기준으로 먼저 검사했으며, 크기 불일치 이미지는 실제 이미지에 좌표를 겹쳐 확인한 후 보정해야 한다.")
    para('이미지 표본','h2')
    selected=samples.groupby(['split','kind_type'],sort=True).head(1)
    cells=[]
    for r in selected.itertuples():
        p=work/r.thumbnail
        if p.exists():
            pic=Image.open(p);pw,ph=pic.size;scale=min(228/pw,145/ph)
            cells.append([RLImage(str(p),width=pw*scale,height=ph*scale),P(f'{r.split} {r.kind_type}\n{Path(r.member).name}','small')])
    display=[]
    for i in range(0,len(cells),2):
        row=cells[i:i+2];display.append(row+['']*(2-len(row)))
    if display:story.append(Table(display,colWidths=[W/2,W/2],style=[('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
    para(f'총 {len(samples)}개 표본은 모두 픽셀 디코딩에 성공했다. 고정 순서로 뽑은 기술 검사용 표본이며 전체 이미지 품질의 대표 표본으로 해석하지 않는다.','small')
    # Missing environment
    title('5 환경 센서의 결측')
    pe=profile[profile.table.eq('environments')].copy()
    table(['필드','전체 연결 행','결측 행','결측률','비결측 고유값'],[[r.field,r.rows,r.missing,f'{r.missing_pct:.2f}%',r.unique_nonmissing] for r in pe.itertuples()],[120,100,100,90,97])
    story.append(heatmap_missing())
    para('원시 연결 행은 같은 센서 시각이 여러 이미지에 반복 연결된 수치다. 위 표는 원시 연결 행을, 아래 농장별 도표는 중복 제거 관측을 분모로 사용한다. 특정 센서가 일부 농장에만 존재하므로 모든 농장을 하나의 평균으로 대치하면 구조적인 결측을 숨길 수 있다.')
    para('rp_value는 관측된 Y와 N으로 이루어진 범주형 필드다. 숫자형 센서와 별도로 집계하고 IQR 검사를 적용하지 않았다. 결측은 자료에 없는 상태이며 N이나 0과 같지 않다.','small')
    # Growth missing
    title('6 생육값 결측과 입력 형식')
    pg=profile[profile.table.eq('growth_index')]
    table(['필드','원시 결측','결측률','숫자 형식 오류','분석 가능 값'],[[r.field,r.missing,f'{r.missing_pct:.2f}%',r.invalid_numeric,r.rows-r.unusable] for r in pg.itertuples()],[145,100,90,90,82])
    para('결측률은 원시 null 빈 문자열 키 부재를 합친 비율이다. 입력 형식 오류는 별도 열로 표시했으며 숫자 기초통계에서는 제외했다. 생육값이 있는 행도 동일 측정일의 값이 여러 이미지에 반복될 수 있어 이미지 수와 독립 생육 측정 수는 다르다.')
    story.append(PageBreak());para('생육값의 입력 형식과 날짜 점검','h1')
    bad=quality[quality.rule.str.startswith('non_numeric_or_nonfinite:')]
    badgroups=bad.groupby(['rule','detail']).size().reset_index(name='n')
    table(['필드','원문 값','발견 행'],[[r.rule.split(':',1)[-1],r.detail,r.n] for r in badgroups.itertuples()],[170,230,107])
    para('과중에 붙은 g를 지우는 작업은 단위 명세를 확인한 후 적용할 수 있다. 백틱은 숫자로 복원할 근거가 없으므로 별도 오류 상태로 유지한다. 원문 보존 열과 정제 숫자 열을 함께 두는 방식이 적절하다. 생육 측정일과 개화일을 사용할 때는 예측 시점 이후 정보를 입력으로 넣지 않도록 확인해야 한다.')
    para('생육 날짜의 시점과 순서 점검','h2')
    gt=s['growth_time']
    table(['날짜 필드','유효 행','가장 이른 날짜','가장 늦은 날짜','촬영일 이후'],[[c,gt[c]['nonmissing'],gt[c]['min'][:10],gt[c]['max'][:10],gt[c]['after_capture']] for c in ['measured_date','bloom1_date','bloom2_date','bloom3_date']],[100,75,120,120,92])
    para(f"bloom2_date가 bloom1_date보다 이른 행 {gt['bloom2_date_before_bloom1_date']:,}건, bloom3_date가 bloom2_date보다 이른 행 {gt['bloom3_date_before_bloom2_date']:,}건이다. 촬영일 이후 개화일은 계획일인지 연도 오입력인지 확인해야 한다. 날짜 순서는 원래 이벤트의 의미를 확인할 검토 후보이며, 전체 원문 날짜와 촬영일은 quality_issues.csv에 기록했다.")
    # Sensor stats
    title('7 환경 센서 수치 분포')
    se=stats[stats.table.eq('environments')]
    table(['필드','유효 관측','최소','Q1','중앙값','Q3','최대','0 개수'],[[r.field,r.n,r.min,r.q1,r.median,r.q3,r.max,r.zero] for r in se.itertuples()],[88,67,60,57,63,57,60,55])
    para(f"통계 기준은 중복 제거 센서 {s['unique_env']:,}건이다. 중복 연결 {s['sensor_repeated_rows']:,}행을 그대로 사용하면 센서보다 촬영 횟수가 많은 구간에 가중치가 생긴다. 동일 시각의 추가 필드 결측 차이 {s['sensor_schema_variant_rows']:,}행은 공동 관측값이 일치해 결합했으며, 공동 관측값의 모순은 {s['sensor_conflicting_rows']:,}행이다.")
    story.append(histpanels([(c,en[c]) for c in ['ti_value','hi_value','ci_value','ir_value','ei_value','pl_value']],320))
    para('히스토그램은 각 변수의 P1부터 P99까지를 표시해 중심 분포를 읽기 쉽게 했고, 그림 아래에 범위 밖 관측 수를 표시했다. 통계와 이상치 검사는 양끝을 포함한 전체 유효 관측을 사용한다. ci_value 최대 5,500과 cl_value 최솟값 -352.4 등은 원시 기록 및 센서 명세와 대조할 후보이다.','small')
    # Extra sensor stats + trends
    title('8 추가 센서와 시간 변화')
    story.append(histpanels([(c,en[c]) for c in ['tl_value','cl_value','hl_value','sr_value','pi_value','el_value']],420))
    para('각 변수의 비결측 관측에서 P1부터 P99까지를 표시했다. cl_value에는 -352.4가 기록되어 있으나 이 값은 중심 표시 범위를 벗어나며 전체 통계 및 이상치 목록에는 포함되어 있다. 센서 코드의 단위를 확인한 후 원문과 비교해야 한다.','small')
    story.append(PageBreak());para('농장별 센서 변화와 수집 간격','h1')
    trend=en.assign(day=pd.to_datetime(en.receive_date,errors='coerce').dt.floor('D')).groupby(['day','farm_id']).ti_value.median().unstack()
    story.append(linechart(trend.reset_index(), 'day',trend.columns.tolist(),220))
    para('아래 선은 농장별 하루 ti_value 중앙값이다. 일별 표본 수와 수집 기간이 농장별로 다르므로 전체 농장의 인과적 생육 효과를 뜻하지 않는다. 시점별 급변은 원시 수신 시각, 센서 교체, 재배 구간 변화와 대조할 필요가 있다.','small')
    cadence=get('cadence')
    table(['농장','고유 시각','간격 중앙값 분','최대 간격 분','60분 초과 간격'],[[r.farm_id,r.unique_times,r.median_gap_minutes,r.max_gap_minutes,r.gaps_gt60min] for r in cadence.itertuples()],[75,98,120,110,104])
    para('간격은 자료에 연결된 센서 기록 사이의 간격이다. 센서 원시 스트림 전체가 아니므로 긴 간격을 실제 센서 중단으로 단정하지 않는다.','small')
    # Growth statistics
    title('9 생육값 수치 분포와 극단값')
    sg=stats[stats.table.eq('growth_index')]
    table(['필드','유효 행','평균','표준편차','최소','중앙값','최대'],[[r.field,r.n,r.mean,r.std,r.min,r.median,r.max] for r in sg.itertuples()],[110,65,65,70,62,65,70])
    story.append(histpanels([(c,gr[c]) for c in GROW_NUM[:6]],300))
    para('stem_length 3,031.5, leaf_length 1,307, leaf_width 88, leaf_cnt 112, stem_thick 135, fr_weight 323은 변수별 원문과 단위를 우선 확인할 극단값이다. 숫자 크기만으로 오류를 확정하지 않았으며 관측 단위와 소수점 입력 가능성을 검토해야 한다. stem_thick처럼 농장에 따라 척도가 달라 보일 수 있는 값은 농장별 분포도 함께 확인한다.')
    # annotation stats
    title('10 객체 클래스와 좌표 분포')
    cc=ann.class_name.replace('', '(빈 이름)').fillna('(null)').value_counts()
    story.append(hbar(cc.index.tolist(),cc.tolist(),180))
    ac=ann.groupby('class_name',dropna=False).agg(objects=('row_key','size'),images=('label_key','nunique'),median_width=('box_width','median'),median_height=('box_height','median'),median_area_ratio=('area_ratio','median')).reset_index()
    table(['클래스','객체 수','포함 이미지','너비 중앙값','높이 중앙값','면적 비율 중앙값'],[[r.class_name or '(빈 이름)',r.objects,r.images,r.median_width,r.median_height,f'{100*r.median_area_ratio:.3f}%'] for r in ac.itertuples()],[72,86,95,82,82,90])
    sa=stats[stats.table.eq('annotations')]
    table(['필드','최소','Q1','중앙값','Q3','최대'],[[r.field]+([f'{v*100:.3f}%' for v in [r.min,r.q1,r.median,r.q3,r.max]] if r.field=='area_ratio' else [r.min,r.q1,r.median,r.q3,r.max]) for r in sa.itertuples()],[90,85,83,83,83,83])
    para('category_id는 각 JSON 내부의 로컬 ID이므로 전역 클래스 번호로 바로 사용하면 안 된다. 각 JSON의 categories에서 id를 name으로 매핑한 후 이름 기준으로 집계했다. 같은 category_id에 여러 클래스 이름이 연결되는 것이 이 구조에서는 오류가 아니다.')
    para(f"클래스 '주'는 {int(cc.get('주',0)):,}개로 매우 드물고 빈 클래스 이름은 {int(cc.get('(빈 이름)',0)):,}개다. 박스 면적은 이미지 너비×높이로 나눈 상대 면적도 함께 계산했다. 음수 면적은 원시 좌표 오류에 따른 값이며 정정 전 분포에 그대로 남겼다.")
    # statistical outlier tables
    title('11 통계적 이상치 탐지 결과')
    para('유효한 숫자에 대해 Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과를 후보로 표시했다. IQR은 Q3 - Q1이다. 유효값 20개 미만 또는 IQR이 0인 그룹에는 이 규칙을 적용하지 않았다. 결측과 숫자 형식 오류는 별도 점검이다.')
    para('전체 기준과 맥락별 기준을 모두 계산했다. 맥락은 이미지의 품종, 센서와 생육값의 농장, 객체 주석의 클래스이다. 센서 임계값은 중복 제거 관측에서 추정한 후 원시 연결 행에 적용했다. 같은 행의 여러 변수가 표시될 수 있으므로 아래 숫자는 서로 더해 고유 이상 이미지 수로 해석하면 안 된다.')
    for i,types in enumerate([['images','environments'],['growth_index','annotations']]):
        if i:story.append(PageBreak());para('생육값과 객체 주석의 통계적 이상치','h1')
        b=outs[outs.table.isin(types)]
        table(['테이블','변수','유효 원시 행','전체 IQR','그룹 IQR','합집합 셀'],[[r.table,r.field,r.raw_nonmissing,r.global_iqr,r.context_iqr,r.union] for r in b.itertuples()],[90,110,80,75,75,77])
    para(f"전체 후보는 {s['statistical_outlier_cells']:,}개 변수 셀이다. 이는 오류 확정 건수가 아니며, x y 좌표나 객체 크기의 꼬리 분포는 드문 위치와 큰 객체를 반영할 수도 있다. 본문의 기초통계는 후보를 삭제하기 전의 원본 기준이다. 전체 및 그룹별 실제 경곗값은 첨부 outlier_bounds.csv에, 모든 후보 값과 행 키는 outlier_records.csv에 수록했다.")
    # quality issues
    title('12 구조 오류와 연결 품질')
    rule_ko={
      'fname_differs_json_stem':'fname과 JSON 이름 차이', 'nearest_sensor_abs_lag_gt15min':'최단 센서 시차 15분 초과',
      'duplicate_image_key:capture_series':'동일 계열 시각 중복 행','label_source_dimension_mismatch':'라벨 실제 이미지 해상도 차이',
      'cross_split_same_image_key:capture_series':'분할 간 동일 계열 시각 중복 행',
      'segmentation_outside_image':'분할 좌표 이미지 경계 초과','bbox_outside_image':'박스 이미지 경계 초과',
      'bbox_nonpositive_size':'박스 너비 또는 높이 0 이하','missing_class_name':'클래스 이름 없음'}
    table(['점검 항목','발견 행','해석'],[[rule_ko.get(r.rule,r.rule),r.records,
      '명명 규칙 차이' if r.rule=='fname_differs_json_stem' else '통계 후보와 별도 점검'] for r in issues.itertuples()],[320,70,117])
    table(['추가 전수 점검','발견'],[
      ['JSON 파싱 실패',count('json_parse_error')],['TAR 읽기 실패',count('tar_read_error')],['이미지 헤더 읽기 실패',count('image_header_error')],
      ['라벨에 대응 원본 없음',s['pairing'].get('left_only',0)],['원본에 대응 라벨 없음',s['pairing'].get('right_only',0)],
      ['객체 image_id 연결 불일치',count('annotation_image_id_mismatch')],['정의되지 않은 category_id',count('unknown_category_id')],
      ['bbox 면적과 area 차이 1% 초과',count('area_bbox_mismatch_gt1pct')],['센서 농장과 이미지 농장 불일치',count('environment_farm_mismatch')],
      ['hi_value 범위 가정 위반',count('domain_range:hi_value')]],[390,117])
    para(f"박스와 분할 좌표의 경계 검사에는 반올림 오차 허용치 0.02픽셀을 적용했다. fname 명명 규칙 차이는 {s['fname_differs_json_stem']:,}건으로 Validation 전체에 나타난다. fname만으로 원본을 찾으면 실패하지만 JSON 파일의 stem을 사용하면 {s['pairing'].get('both',0):,}쌍 모두 연결된다.")
    # examples
    title('13 확인할 오류 사례')
    examples=quality[quality.rule.isin(['bbox_nonpositive_size','bbox_outside_image','segmentation_outside_image'])].groupby('rule',sort=False).head(2)
    table(['규칙과 파일 위치','원문 좌표 또는 설명'],[[r.rule+'\n'+r.row_key,r.detail] for r in examples.itertuples()],[245,262])
    dm=pd.read_csv(audit/'dimension_mismatches.csv')
    table(['라벨 파일','라벨 해상도','실제 해상도'],[[str(r.member_label),f'{fmt(r.width_label)} × {fmt(r.height_label)}',f'{fmt(r.width_source)} × {fmt(r.height_source)}'] for r in dm.head(5).itertuples()],[285,111,111])
    para('좌표가 이미지 밖에 있으면 무조건 자르기 전에 실제 해상도와 라벨 해상도를 먼저 맞춰야 한다. 음수 너비나 높이도 좌상단 좌표 재정의 또는 재라벨링 여부를 결정한 뒤 고친다. 동일 원본을 화면에 띄워 좌표를 겹치는 확인 절차가 필요하다. 이 보고서는 좌표 오류를 찾았으며 자동 보정은 수행하지 않았다.')
    # alignment
    title('14 이미지와 센서의 시각 정합성')
    lag=s['lag']
    table(['항목','결과'],[['센서 연결 이미지',lag['n']],['가장 가까운 센서와 시차 중앙값 분',lag['median']],['절대 시차 P95 분',lag['p95']],['최대 절대 시차 분',lag['max_abs']],
      ['15분 초과 이미지',lag['gt15']],['60분 초과 이미지',lag['gt60']],['240분 초과 이미지',lag['gt240']],['가장 가까운 센서가 촬영 이후인 이미지',lag['after_capture']]],[360,147])
    laggroups=pd.cut(near.abs_lag,bins=[-np.inf,1,5,15,60,240,np.inf],labels=['1분 이하','1분 초과 5분 이하','5분 초과 15분 이하','15분 초과 60분 이하','60분 초과 240분 이하','240분 초과']).value_counts(sort=False)
    story.append(hbar(laggroups.index.astype(str).tolist(),laggroups.tolist(),180))
    lagfarm=near.groupby('farm_id').agg(images=('label_key','size'),median=('abs_lag','median'),p95=('abs_lag',lambda x:x.quantile(.95)),max=('abs_lag','max'),gt15=('abs_lag',lambda x:int(x.gt(15).sum()))).reset_index()
    table(['농장','이미지','중앙값 분','P95 분','최대 분','15분 초과'],[[r.farm_id,r.images,r.median,r.p95,r.max,r.gt15] for r in lagfarm.itertuples()],[75,90,85,85,87,85])
    para('각 이미지에 연결된 두 센서 중 촬영 시각과 절대 차이가 가장 작은 관측을 사용했다. 15분은 본 EDA의 검토 기준이며 센서 명세의 공식 허용치가 아니다. 극단적인 지연 이미지는 센서 기반 분석에서 별도 플래그로 보존하고 허용 시차를 정한 뒤 포함 여부를 결정한다.')
    # leakage and correlations
    title('15 중복과 학습 검증 분할')
    table(['동일성 기준','중복 행','중복 키 그룹'],[[r['key'],r['duplicate_rows'],r['duplicate_groups']] for r in s['duplicates']],[260,120,127])
    table(['분할 간 공유 기준','공유 키','영향 행'],[[r['key'],r['shared_keys'],r['affected_rows']] for r in s['leakage']],[260,120,127])
    para(f"이미지 ID와 JSON 파일명 중복은 없고 원문 JSON SHA256 중복도 {s['json_sha_duplicate_rows']:,}행이다. 다만 동일 촬영 계열과 시각이 Training과 Validation에 함께 존재하는 사례가 있다. 촬영 계열은 JSON 파일명에서 마지막 숫자 ID를 제거한 접두사로 정의한 기술적 그룹이며 개별 식물 ID라고 단정하지 않는다.")
    para('동일 농장과 수집 기간에서 이미지 단위로 나누면 인접 프레임과 센서값을 학습과 검증에서 공유할 수 있다. 새 농장에 대한 성능이 목적이면 농장 단위 홀드아웃을, 미래 시점 예측이 목적이면 시간 순 분할과 충분한 간격을 적용해야 한다. 동일 이미지 ID가 없다는 사실만으로 시각적 중복이나 누수 가능성이 사라지는 것은 아니다.')
    para('센서 상관을 볼 때의 주의','h2')
    corr=get('correlation');pairs=[]
    for i,c in enumerate(corr.columns):
        for j in range(i+1,len(corr.columns)):
            other=corr.columns[j];v=corr.iloc[i,j]
            n=int(en[[c,other]].notna().all(axis=1).sum())
            if pd.notna(v):pairs.append((c,other,v,n))
    pairs=sorted(pairs,key=lambda z:abs(z[2]),reverse=True)
    table(['변수 1','변수 2','Pearson r','쌍별 유효 관측'],pairs[:8],[130,130,100,147])
    para('상관은 중복 제거 관측의 쌍별 완전 사례로 계산했고 최소 30개 관측을 요구했다. 변수마다 관측 농장과 시간 구간이 다르며 농장 효과와 계절 효과가 섞여 있다. 이 값은 인과관계 또는 생육 개선 효과의 증거가 아니다.','small')
    # general fields and flags
    title('16 이미지와 공통 메타데이터의 결측')
    others=profile[~profile.table.isin(['growth_index','environments']) & profile.missing.gt(0)]
    table(['테이블','필드','전체 행','원시 결측','결측률'],[[r.table,r.field,r.rows,r.missing,f'{r.missing_pct:.3f}%'] for r in others.itertuples()],[97,130,100,90,90])
    para('결측이 발견된 필드만 표시했고 결측 0건인 필드는 부록에 모두 수록했다. images의 plant_body와 stem flower fruit grpoint는 존재하는 스키마 구간이 달라 부재를 자동으로 0으로 채우지 않았다. etc_infor의 빈 create_date inform과 licenses의 빈 url은 설명 메타데이터의 결측이며 센서나 생육값 결측과 구분해서 처리한다.')
    bi=stats[stats.table.eq('images') & ~stats.field.isin(['width','height'])]
    table(['이진 플래그','관측 행','0','1','결측'],[[r.field,r.n,r.zero,r.n-r.zero,r.missing_baseline] for r in bi.itertuples()],[115,98,98,98,98])
    # actions and execution
    title('17 정제 우선순위와 재현 방법')
    para('먼저 이름이 다른 필드를 정규화하고 JSON stem 기준의 원본 연결을 유지한다. categories의 로컬 ID를 클래스 이름으로 변환한 다음, 클래스 사전을 별도로 고정한다. 빈 클래스 이름 5건과 희소 클래스 주 61건은 라벨 의도를 확인한다.')
    para('다음으로 해상도 불일치와 음수 박스 및 경계 밖 좌표를 원본 이미지 위에서 확인한다. 해상도를 바꾸거나 박스를 잘라내는 자동 처리 전에 실제 좌표계가 어느 이미지를 가리키는지 확인해야 한다. 형식 오류는 원문을 보존하고 숫자 파싱 상태를 별도 열로 기록한다.')
    para('결측은 농장별 센서 가용성과 생육 측정일을 기준으로 다룬다. 미래 관측으로 과거 결측을 메우지 않도록 하고 대치 여부와 관측 여부 플래그를 함께 저장한다. 통계적 이상치는 원시 검토 후 보존 보정 제외 중 적절한 처리를 결정한다. 드문 값이라는 이유만으로 일괄 제거하지 않는다.')
    para('마지막으로 목표 평가 상황에 맞게 농장 촬영 계열 시간 구간을 분리하고, 허용 시차를 넘는 센서 연결을 표시한다. 원시 이미지 기준 성능 외에 농장 품종 생육 단계별 성능도 함께 비교한다.')
    para('실행 방법','h2')
    para('필요 패키지  numpy pandas Pillow reportlab pypdf\n권장 Python 3.11 이상\n실행  python eda_strawberry.py --root . --out eda_results\n캐시 재사용  python eda_strawberry.py --root . --out eda_results --reuse\n보고서만 재작성  python eda_strawberry.py --out eda_results --report-only')
    para('기본 글꼴 경로는 Windows의 맑은 고딕이다. 다른 운영체제에서는 --font와 --font-bold에 한글을 지원하는 TTF 경로를 지정한다. --reuse는 폴더의 원본이 바뀌지 않았을 때만 사용한다. 원본을 바꾼 뒤에는 캐시 없이 전수 스캔을 다시 실행해야 한다.')
    table(['첨부 파일','내용'],[
      ['missing_records.csv','결측이 있는 모든 정규화 행과 필드 목록'],['outlier_records.csv','전체 통계 후보의 실제 값 전체 및 그룹 판정과 경곗값'],
      ['quality_issues.csv','모든 형식 좌표 연결 시각 중복 점검 행과 원문 설명'],['raw_schema.csv / missing_summary.csv','원문 키 부재 null 빈 값과 정규화 결측 요약'],
      ['numeric_summary.csv / outlier_bounds.csv','모든 수치 통계와 전체 및 그룹별 임계값'],['image_metadata.csv','전수 이미지 메타데이터 및 촬영 계열'],
      ['unique_environment_observations.csv','중복 제거 센서값 전체'],['archive_inventory.csv','612개 TAR별 파일 수와 용량'],
      ['그 외 CSV와 README.txt','연결 불일치 시차 분포 표본 검사 클래스 매핑 등']],[190,317])
    para('행 키는 split/TAR/압축 내부 파일#테이블:순번 형식이다. 순번은 0부터 시작하며 누락된 셀 목록은 | 기호로 구분한다. outlier_records의 한 행은 하나의 변수 셀이고 quality_issues의 한 행은 하나의 규칙 위반이므로 중복될 수 있다.','small')
    # exhaustive raw schema appendix
    title('부록 원문 필드별 결측 전수 목록')
    para('정규화 전 원문 스키마 기준이다. absent는 해당 키가 없는 경우, null은 키가 있으나 값이 null인 경우, blank는 빈 문자열, empty는 빈 컨테이너다. root의 environments와 envrionments 및 corps_id와 crops_id는 별칭이므로 이 부록의 부재를 합쳐 정보 손실로 해석하지 않는다. 원시 결측과 형식 오류를 분리한 분석용 결측은 앞 절과 missing_summary.csv를 참고한다.','small')
    for t,b in schema.groupby('table',sort=False):
        para(f'{t}  전체 {int(b.rows.iloc[0]):,}행','h2')
        table(['원문 필드','키 부재','null','빈 문자열','empty','총 결측률'],[[r.field,r.absent,r.null,r.blank,r.empty_container,f'{r.missing_pct:.3f}%'] for r in b.itertuples()],[147,78,78,70,55,79],compact=True)
    # Capture reproducibility metadata before the exact audit is attached.
    date=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
    readme=f'''딸기 수직농장 EDA 전수 감사 목록\n생성 시각 KST: {date}\n코드 버전: {VERSION}\n원본: {root}\n원본은 수정하지 않았습니다.\n\nrow_key = split/TAR/압축내부파일#table:0부터 시작하는 순번\nmissing_records.csv: 원시 결측(키 부재/null/빈 문자열) 필드 목록. 잘못된 숫자는 quality_issues에 별도 기록.\noutlier_records.csv: 1행=1개 변수 셀. global/context IQR 합집합. 오류 확정 아님.\nquality_issues.csv: 1행=1개 규칙. 같은 관측이 여러 규칙에 포함 가능.\nraw_schema.csv: 원문 키 부재/null/빈 문자열 구분. 별칭 부재는 실제 정보 결측과 다름.\nmissing_summary.csv: 별칭 정규화 후 결측 및 숫자 형식 오류.\nunique_environment_observations.csv: farm_id+receive_date 기준 중복 제거. 공동 관측값이 같고 추가 필드만 빠진 경우 결합. 모순값은 유지.\n생육 통계는 이미지에 연결된 행 단위이며 독립 생육 측정 수와 다릅니다.\n모든 원본 JPEG 헤더 검사. 전체 픽셀 디코딩/해시 전수 검사 아님. 12장 고정 순서 표본 디코딩.\n센서 코드 단위 공식 명세가 없어 원문 코드와 값을 유지합니다.\nrequirements: numpy pandas Pillow reportlab pypdf\n재실행: python eda_strawberry.py --root . --out eda_results\n캐시 재사용은 원본 불변일 때만 --reuse 옵션 사용.\n'''
    (audit/'README.txt').write_text(readme,encoding='utf-8-sig')
    (audit/'analysis_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    from importlib.metadata import version
    versions={n:version(n) for n in ['numpy','pandas','Pillow','reportlab','pypdf']}
    (audit/'environment.json').write_text(json.dumps(dict(created=date,python=sys.version,packages=versions,code_sha256=sha(Path(__file__).read_bytes())),ensure_ascii=False,indent=2),encoding='utf-8')
    audit_zip=out/'eda_audit.zip'
    with zipfile.ZipFile(audit_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(audit.iterdir()):z.write(p,p.name)
    base=work/'report_body.pdf';final=out/'strawberry_eda_report.pdf'
    class ReportDoc(SimpleDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name in ['title','h1']:
                key='section_'+str(len(getattr(self,'_outline',[])))
                self._outline=getattr(self,'_outline',[])+[key]
                self.canv.bookmarkPage(key);self.canv.addOutlineEntry(flowable.getPlainText(),key,level=0,closed=False)
    doc=ReportDoc(str(base),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=43,
                          title='딸기 수직농장 데이터 EDA 결과 보고서',author='Data Analysis',pageCompression=1)
    def page(c,doc):
        c.saveState();c.setFont('MG',7);c.setFillColor(gray)
        c.drawString(44,24,'딸기 수직농장 데이터 EDA | '+analysis_date);c.drawRightString(A4[0]-44,24,str(doc.page));c.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)
    writer=PdfWriter();writer.clone_document_from_reader(PdfReader(base));writer.add_attachment('eda_audit.zip',audit_zip.read_bytes())
    writer.add_metadata({'/Title':'딸기 수직농장 데이터 EDA 결과 보고서','/Author':'Data Analysis','/Subject':'전수 EDA 결측치 이상치 품질 분석 및 전체 감사 CSV 첨부'})
    with final.open('wb') as f:writer.write(f)
    verify=PdfReader(final)
    assert len(verify.pages)>0 and 'eda_audit.zip' in verify.attachments
    assert sha(verify.attachments['eda_audit.zip'][0])==sha(audit_zip.read_bytes())
    log('report',len(verify.pages),'pages',str(final),'audit MB',round(audit_zip.stat().st_size/1e6,2))

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('eda_results'))
    ap.add_argument('--reuse',action='store_true');ap.add_argument('--scan-only',action='store_true')
    ap.add_argument('--report-only',action='store_true');ap.add_argument('--analyze-only',action='store_true')
    ap.add_argument('--font',default='C:/Windows/Fonts/malgun.ttf');ap.add_argument('--font-bold',default='C:/Windows/Fonts/malgunbd.ttf')
    args=ap.parse_args();root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    try:sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:pass
    if args.report_only:build_report(root,out,args);return
    if not args.reuse:scan(root,out)
    if not args.scan_only:
        analyze(root,out)
        if not args.analyze_only:build_report(root,out,args)

if __name__=='__main__':main()
