from pycocotools.coco import COCO
import numpy as np

## for the preprocessing code

## Image preprocessing

## caption preprocessing

def get_data(annFile):
    """
    Given annotation file path, return array of images and array
    of captions corresponding to these images (as strings)
    """ # http://images.cocodataset.org/zips/train2014.zip
    #   http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    # For Test:::
    #### http://images.cocodataset.org/zips/test2015.zip
    #### http://images.cocodataset.org/annotations/image_info_test2015.zip
    
    coco = COCO(annFile)
    anns = coco.loadAnns(coco.getAnnIds())
    img_ids = map(lambda x: x.image_id, anns)
    captions = map(lambda x: x.caption, anns)
    imgs = coco.loadImgs(img_ids)

    return imgs, captions

