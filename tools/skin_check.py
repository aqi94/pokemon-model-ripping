"""Face vs body skin colour per converted trainer, sampled from the baked base colour at the UVs of the
vertices that use each skin material (lighting-independent). Flags G/R differences > 0.08. body_skin can
also cover gloves/tights, so confirm flags against renders (tools/render_models.mjs).

    ./.venv/Scripts/python.exe tools/skin_check.py
"""
import json,struct,io,glob,os,sys
import numpy as np
from PIL import Image
sys.path.insert(0,'.')
from pipeline.importers.za.importer import srgb_to_linear as s2l, linear_to_srgb as l2s
names={m['id']:m['displayName'] for m in json.load(open('out/index.json'))['models']}
CT={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
def load(path):
    b=open(path,'rb').read(); n=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+n]); b0=20+n+8
    def acc(i):
        a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]; o=b0+bv.get('byteOffset',0)+a.get('byteOffset',0)
        return np.frombuffer(b,CT[a['componentType']],a['count']*NC[a['type']],o).reshape(a['count'],NC[a['type']])
    cache={}
    def img(ti):
        if ti not in cache:
            im=j['images'][j['textures'][ti]['source']]; bv=j['bufferViews'][im['bufferView']]; o=b0+bv.get('byteOffset',0)
            cache[ti]=np.asarray(Image.open(io.BytesIO(b[o:o+bv['byteLength']])).convert('RGBA')).astype(np.float32)/255
        return cache[ti]
    return j,acc,img
def skin(j,acc,img,names_):
    mids=[i for i,m in enumerate(j['materials']) if m['name'] in names_]
    if not mids: return None
    out=[]
    for mesh in j['meshes']:
        for p in mesh['primitives']:
            if p.get('material') not in mids: continue
            m=j['materials'][p['material']]; pb=m['pbrMetallicRoughness']
            if 'baseColorTexture' not in pb: continue
            t=img(pb['baseColorTexture']['index']); f=np.array(pb.get('baseColorFactor',[1,1,1,1])[:3])
            uv=acc(p['attributes']['TEXCOORD_0']).astype(np.float64)
            h,w=t.shape[:2]
            x=np.clip((np.mod(uv[:,0],1)*w).astype(int),0,w-1); y=np.clip((np.mod(uv[:,1],1)*h).astype(int),0,h-1)
            px=t[y,x]; px=px[px[:,3]>0.5][:,:3]
            out.append(s2l(px)*f)
    if not out: return None
    lin=np.concatenate(out); lum=lin@[.2126,.7152,.0722]
    sel=lin[(lum>np.percentile(lum,30))&(lum<np.percentile(lum,97))]
    return l2s(np.median(sel,0))*255
print(f"{'id':10} {'name':14} {'face':>15} {'body skin':>15}  G/R face body")
for p in sorted(glob.glob('out/za/tr*/model.glb')):
    mid=os.path.basename(os.path.dirname(p))
    j,acc,img=load(p)
    f=skin(j,acc,img,{'face_skin'}); b=skin(j,acc,img,{'body_skin','hairstyle_skin'})
    if f is None or b is None:
        print(f'{mid:10} {names.get(mid,""):14} no face_skin/body_skin material ({[m["name"] for m in j["materials"] if "@" not in m["name"]][:8]})'); continue
    gf,gb=f[1]/f[0],b[1]/b[0]
    print(f"{mid:10} {names.get(mid,''):14} {str(f.round().astype(int)):>15} {str(b.round().astype(int)):>15}  {gf:.2f} {gb:.2f}{'  <-- mismatch' if abs(gf-gb)>0.08 else ''}")
