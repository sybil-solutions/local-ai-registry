"""Group tensor bytes of an MLX safetensors checkpoint by component (reads headers only)."""
import json, struct, sys, glob, re, collections
d=sys.argv[1]; g=collections.Counter(); per_layer=collections.Counter(); shapes={}
for f in sorted(glob.glob(d+"/*.safetensors")):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n))
    for k,v in h.items():
        if k=="__metadata__": continue
        b=v["data_offsets"][1]-v["data_offsets"][0]
        m=re.search(r"layers\.(\d+)\.",k); L=int(m.group(1)) if m else None
        if "vision" in k: c="vision"
        elif "lm_head" in k: c="lm_head"
        elif "embed_tokens" in k: c="embed"
        elif ".experts." in k: c="experts"
        elif L is not None: c="layer_other"
        else: c="other"
        g[c]+=b
        if L is not None: per_layer[(L,c)]+=b
        if L==6: shapes[k]=(v["shape"],v["dtype"],b)
for c,b in g.most_common(): print(f"{c:12s} {b/1e9:8.3f} GB")
print("layer6 tensors:"); [print(" ",k,*v) for k,v in sorted(shapes.items())]
print("expert GB/layer L0,L6,L20:",[round(per_layer[(L,'experts')]/1e9,3) for L in (0,6,20)], "other GB/layer:",[round(per_layer[(L,'layer_other')]/1e9,4) for L in (0,5,6,20)])
