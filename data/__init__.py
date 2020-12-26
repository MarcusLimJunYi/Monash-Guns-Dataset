# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .coco import COCODetection
from .mgd0712 import MGDDetection, AnnotationTransform, detection_collate, MGD_CLASSES
from .data_augment import *
from .anchors import *
