import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
## for the preprocessing code

## Image preprocessing

## caption preprocessing

def get_data():
    """
    Given annotation file path, return array of images and array
    of captions corresponding to these images (as strings)
    """ 
    folder = "CUB_200_2011"
    classes_txt = open(folder + "/classes.txt")
    classes = classes_txt.read()
    classes_txt.close()
    classnames = classes.splitlines()
    classnames = list(map(lambda x: x[x.find(' ') + 1:], classnames))
    
    img_class_labels_txt = open(folder + "/image_class_labels.txt")
    img_class_labels = img_class_labels_txt.read()
    img_class_labels_txt.close()
    
    img_class_labels = img_class_labels.splitlines()
    
    img_attributes = []
    attrib_list = open(folder + "/attributes/image_attribute_labels.txt")
    img_attribs = attrib_list.read()
    attrib_list.close()
    img_attribs = img_attribs.splitlines()
    for i in range(0, 11788):
        img_attributes.append(list(map(lambda x: int(x[x.find(' ', x.find(' ')+1)+1:x.find(' ', x.find(' ')+1)+2]), img_attribs[i:i+312])))
    print(tf.convert_to_tensor(img_attributes))
    
    images = np.zeros((11788, 64, 64, 3))
    image_list = open(folder + "/images.txt")
    image_ids = image_list.read()
    image_list.close()
    image_ids = image_ids.splitlines()
    
    for i in range(0, 11788):
        print(i)
        img_data = Image.open(folder + "/images/" + image_ids[i][image_ids[i].find(' ')+1:], 'r')
        #print(np.array(img_data).shape)
        images[i] = resize(np.asarray(img_data), (64, 64, 3))
        
    print(0)
    return images, np.array(img_attributes)



#def main():
    #get_data()
    
#main()