"""Regression coverage proving production analysis is not keyed to samples/filenames."""
from pathlib import Path
from tests.drawing import canvas, shape, arrow, encode
from backend.pipeline import analyze

def custom_user_flowchart():
    image=canvas(1200,1400)
    shape(image,'terminal',(420,40,360,90),'Start')
    shape(image,'input',(410,240,380,110),'Input temperature:float')
    shape(image,'process',(410,470,380,110),'fan = temperature > 32')
    shape(image,'output',(410,700,380,110),'Output fan')
    shape(image,'terminal',(420,930,360,90),'End')
    arrow(image,[(600,130),(600,240)])
    arrow(image,[(600,350),(600,470)])
    arrow(image,[(600,580),(600,700)])
    arrow(image,[(600,810),(600,930)])
    return image

def test_arbitrary_user_flowchart_is_analyzed(settings):
    data=encode(custom_user_flowchart())
    ctx=analyze(data,'my-home-automation-diagram.png',settings)
    assert not ctx.has_errors,[d.to_dict() for d in ctx.diagnostics]
    assert [node.text for node in ctx.elements][:3]==['Start','Input temperature:float','fan = temperature > 32']
    assert ctx.pseudocode and ctx.ir is not None
    assert all(node.id not in {'sample','sequence.png'} for node in ctx.elements)
