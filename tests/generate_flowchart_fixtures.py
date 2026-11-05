import cv2, json, os
import numpy as np
root='/mnt/data/modulo_work/modulo_launcher/tests/fixtures/flowcharts'
os.makedirs(root,exist_ok=True)
# Shared metadata is intentionally explicit; image coordinates are hand-authored synthetic ground truth.
cases={
'simple_sequence': {'tags':['sequence'], 'nodes':[('terminal','Start'),('process','x = 1'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None)]},
'input_output': {'tags':['input_output'], 'nodes':[('terminal','Start'),('data','Input n'),('data','Print n'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None),('n3','n4',None)]},
'decision': {'tags':['decision','branch_labels'], 'nodes':[('terminal','Start'),('data','Input n'),('decision','n > 0?'),('process','Print Positive'),('process','Print Other'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None),('n3','n4','Yes'),('n3','n5','No'),('n4','n6',None),('n5','n6',None)]},
'nested_branches': {'tags':['decision','nested_branches'], 'nodes':[('terminal','Start'),('decision','A?'),('decision','B?'),('process','One'),('process','Two'),('process','Three'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3','Yes'),('n2','n6','No'),('n3','n4','Yes'),('n3','n5','No'),('n4','n7',None),('n5','n7',None),('n6','n7',None)]},
'loop': {'tags':['loop','decision'], 'nodes':[('terminal','Start'),('process','i = 0'),('decision','i < 3?'),('process','i = i + 1'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None),('n3','n4','Yes'),('n4','n3',None),('n3','n5','No')]},
'crossings': {'tags':['crossing','connectors'], 'nodes':[('terminal','Start'),('process','Left'),('process','Right'),('terminal','End')], 'edges':[('n1','n2',None),('n1','n3',None),('n2','n4',None),('n3','n4',None)]},
'long_connectors': {'tags':['long_connectors'], 'nodes':[('terminal','Start'),('process','Do work'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None)]},
'small_text': {'tags':['small_text'], 'nodes':[('terminal','Start'),('process','total = total + 1'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None)]},
'math_operators': {'tags':['math_operators'], 'nodes':[('terminal','Start'),('data','Input a, b'),('process','c = a * b + 2'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None),('n3','n4',None)]},
'colored_background': {'tags':['colored_background'], 'nodes':[('terminal','Start'),('process','Count = 0'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None)]},
'low_resolution': {'tags':['low_resolution'], 'nodes':[('terminal','Start'),('process','x = x + 1'),('terminal','End')], 'edges':[('n1','n2',None),('n2','n3',None)]},
}

def draw(name,case):
    scale=0.5 if name=='low_resolution' else 1
    W,H=(900,760) if name=='crossings' else (800,900)
    im=np.full((H,W,3),255,np.uint8)
    if name=='colored_background': im[:]= (245,250,255)
    # node positions by case, otherwise vertical sequence; branch-specific layouts
    n=len(case['nodes'])
    if name in ('decision','nested_branches'):
        pos={'n1':(400,65),'n2':(400,180),'n3':(400,330),'n4':(190,480),'n5':(610,480),'n6':(400,650),'n7':(400,760)}
        if name=='decision': pos={'n1':(400,55),'n2':(400,160),'n3':(400,285),'n4':(210,465),'n5':(590,465),'n6':(400,650)}
    elif name=='crossings': pos={'n1':(450,70),'n2':(220,360),'n3':(680,360),'n4':(450,650)}
    elif name=='loop': pos={'n1':(400,55),'n2':(400,170),'n3':(400,320),'n4':(610,480),'n5':(400,650)}
    else:
        ys=np.linspace(70, H-100,n).astype(int); pos={f'n{i+1}':(400,int(y)) for i,y in enumerate(ys)}
    # edge routes use simple orthogonal paths with arrowheads
    def center(nid): return pos[nid]
    for si,(a,b,label) in enumerate(case['edges']):
        x1,y1=center(a); x2,y2=center(b)
        typ_a=case['nodes'][int(a[1:])-1][0]; typ_b=case['nodes'][int(b[1:])-1][0]
        # node extents approx; draw routes outside shapes
        if name=='loop' and a=='n4' and b=='n3':
            pts=[(x1+95,y1),(740,y1),(740,y2),(x2+90,y2)]
        elif name in ('decision','nested_branches') and a in ('n3','n2') and abs(x2-x1)>20:
            pts=[(x1,y1+55),(x1,y1+85),(x2,y1+85),(x2,y2-50)]
        elif name=='crossings':
            pts=[(x1,y1+40),(x1,220),(x2,220),(x2,y2-40)]
        elif abs(x2-x1)<5:
            pts=[(x1,y1+45),(x2,y2-45)]
        else:
            mid=(y1+y2)//2
            pts=[(x1,y1+45),(x1,mid),(x2,mid),(x2,y2-45)]
        for p,q in zip(pts,pts[1:]): cv2.line(im,p,q,(25,25,25),2,cv2.LINE_AA)
        # arrowhead at last endpoint
        end=pts[-1]; prev=pts[-2]; v=np.array(end)-np.array(prev); norm=np.linalg.norm(v) or 1; u=v/norm; perp=np.array([-u[1],u[0]])
        tip=np.array(end); base=tip-u*13
        cv2.line(im,tuple(tip.astype(int)),tuple((base+perp*7).astype(int)),(25,25,25),2,cv2.LINE_AA)
        cv2.line(im,tuple(tip.astype(int)),tuple((base-perp*7).astype(int)),(25,25,25),2,cv2.LINE_AA)
        if label:
            lx=(pts[-2][0]+pts[-1][0])//2+8; ly=(pts[-2][1]+pts[-1][1])//2
            cv2.putText(im,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,.48,(10,10,10),1,cv2.LINE_AA)
    for i,(kind,text) in enumerate(case['nodes'],1):
        x,y=pos[f'n{i}']; color=(255,255,255)
        if name=='colored_background': color=(205,235,255)
        if kind=='decision': pts=np.array([[x,y-52],[x+100,y],[x,y+52],[x-100,y]],np.int32); cv2.fillPoly(im,[pts],color); cv2.polylines(im,[pts],True,(20,20,20),2,cv2.LINE_AA)
        elif kind=='data':
            pts=np.array([[x-100,y-35],[x+85,y-35],[x+100,y+35],[x-85,y+35]],np.int32); cv2.fillPoly(im,[pts],color); cv2.polylines(im,[pts],True,(20,20,20),2,cv2.LINE_AA)
        elif kind=='terminal':
            cv2.ellipse(im,(x,y),(100,37),0,0,360,color,-1,cv2.LINE_AA)
            cv2.ellipse(im,(x,y),(100,37),0,0,360,(20,20,20),2,cv2.LINE_AA)
        else:
            cv2.rectangle(im,(x-110,y-40),(x+110,y+40),color,-1)
            cv2.rectangle(im,(x-110,y-40),(x+110,y+40),(20,20,20),2,cv2.LINE_AA)
        font=.55 if name=='small_text' else .65
        if name=='low_resolution': font=.55
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,font,1)
        cv2.putText(im,text,(x-tw//2,y+th//2),(cv2.FONT_HERSHEY_SIMPLEX),font,(0,0,0),1,cv2.LINE_AA)
    if scale<1: im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(root,name+'.png'),im)
    return {'image':name+'.png','tags':case['tags'],'nodes':[{'id':f'n{i+1}','type':t,'text':txt} for i,(t,txt) in enumerate(case['nodes'])], 'edges':[{'source':a,'target':b,'label':l} for a,b,l in case['edges']], 'expected_pseudocode': 'See expected/'+name+'.txt'}
manifest={k:draw(k,v) for k,v in cases.items()}
with open('/mnt/data/modulo_work/modulo_launcher/tests/fixtures/expected/manifest.json','w') as f: json.dump(manifest,f,indent=2)
for k,v in cases.items():
    with open('/mnt/data/modulo_work/modulo_launcher/tests/fixtures/expected/'+k+'.txt','w') as f:
        f.write('# Expected control-flow specification (not asserted as OCR output until baseline is reviewed)\n')
        f.write('\n'.join(f"{i+1}. {kind}: {text}" for i,(kind,text) in enumerate(v['nodes']))+'\n\n')
        f.write('Directed edges:\n'+'\n'.join(f'{a} -> {b}'+(f' [{l}]' if l else '') for a,b,l in v['edges'])+'\n')
