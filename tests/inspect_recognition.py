"""Developer diagnostic for raster fixture recognition (not application logic)."""
import argparse
import cv2
import numpy as np
from backend.config import Settings
from backend.models import PipelineContext
from backend.services.intake_service import IntakeService
from backend.services.vision_service import VisionService, shape_contours, classify_shape
from backend.services.graph_service import contacts
from tests.drawing import encode, sequence, branch, loop, independent_pdf

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['sequence','horizontal','branch','loop','pdf']);args=parser.parse_args()
    builders={'sequence':lambda:encode(sequence()),'horizontal':lambda:encode(sequence(horizontal=True)), 'branch':lambda:encode(branch()),'loop':lambda:encode(loop()),'pdf':independent_pdf}
    ctx=IntakeService().run(PipelineContext(filename='diagnostic.pdf' if args.kind=='pdf' else 'diagnostic.png',uploaded_bytes=builders[args.kind](),settings=Settings()))
    for contour in shape_contours(ctx.binary):
        perimeter=cv2.arcLength(contour,True);box=cv2.boundingRect(contour)
        print('GEOMETRY',box,'area ratio',cv2.contourArea(contour)/(box[2]*box[3]),'coarse',cv2.approxPolyDP(contour,.025*perimeter,True).reshape(-1,2).tolist(),'fine',len(cv2.approxPolyDP(contour,.01*perimeter,True)))
    VisionService().run(ctx)
    print('NODES',[(n.id,n.type,n.text,n.bbox.to_list()) for n in ctx.elements])
    print('WORDS',[(w.text,w.confidence,w.bbox.to_list()) for w in ctx.edge_words])
    count,labels,stats,_=cv2.connectedComponentsWithStats(ctx.stroke_mask,8)
    for i in range(1,count):
        if stats[i,4]<20:continue
        y,x=np.where(labels==i);points=np.column_stack((x,y))
        print('STROKE',stats[i].tolist(),'PORTS',[(p.node.id,p.point,p.head) for p in contacts(points,ctx.elements,ctx.stroke_mask.shape,ctx.settings.ARROW_ATTACH_DISTANCE)])
    cv2.imwrite('/tmp/modulo-'+args.kind+'-strokes.png',255-ctx.stroke_mask)
