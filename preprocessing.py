from pycocotools.coco import COCO
import numpy as np

## for the preprocessing code

## Image preprocessing

## caption preprocessing

def get_data(annFile):
    """
    Given annotation file path, return array of images and array
    of captions corresponding to these images (as strings)
    """
    coco = COCO(annFile)
    anns = coco.loadAnns(coco.getAnnIds())
    img_ids = map(lambda x: x.image_id, anns)
    captions = map(lambda x: x.caption, anns)
    imgs = coco.loadImgs(img_ids)

    return imgs, captions

