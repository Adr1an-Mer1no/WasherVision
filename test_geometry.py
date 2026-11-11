import numpy as np
from vision_utils import WASHERS,box_centroid,calibrate,pixel_to_workspace,bbox_iou,overlaps

def test_centroid():
    assert box_centroid([10,20,30,60])==(20,40)

def test_homography():
    H=calibrate([[0,0],[100,0],[100,100],[0,100]],200,150)
    assert np.allclose(pixel_to_workspace(50,50,H),(100,75))

def test_overlap():
    assert bbox_iou([0,0,10,10],[20,20,30,30])==0
    assert overlaps([0,0,10,10],[[0,0,10,10],[5,5,15,15]],0)

def test_classes():
    assert WASHERS['N1']['material']=='nylon'
    assert WASHERS['W3']['od_mm']==20
