import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.core import *
from utils.adaptive_nms import *
from utils.pycocotools.coco import COCO
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='configs/m2det512_resnet101.py', type=str)
parser.add_argument('-f', '--directory', default='demos/', help='the path to demo images')
parser.add_argument('-m', '--trained_model', default='weights/MGD_GIoU_FL_FFMv3.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--cam', default=-1, type=int, help='camera device id')
parser.add_argument('--show', action='store_true', help='Whether to display the images')
parser.add_argument('--record', action='store_true', help='Record webcam detection video')
parser.add_argument('--log', type=bool, default=False, help='Log detection score for each frame')
parser.add_argument('--ad_nms', type=bool, default=False, help='Enable Adaptive NMS')
args = parser.parse_args()

print_info(' ----------------------------------------------------------------------\n'
           '|                       M2Det Demo Program                             |\n'
           ' ----------------------------------------------------------------------', ['yellow','bold'])

global cfg
cfg = Config.fromfile(args.config)
anchor_config = anchors(cfg)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
net = build_net('test',
                size = cfg.model.input_size,
                config = cfg.model.m2det_config)
init_net(net, cfg, args.trained_model)
print_info('===> Finished constructing and loading model',['yellow','bold'])
net.eval()
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
_preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]
#cats = [_.strip().split(',')[-1] for _ in open('data/coco_labels.txt','r').readlines()]
cats = [_.strip().split(',')[-1] for _ in open('data/MGD_labels.txt','r').readlines()]
labels = tuple(['__background__'] + cats)

def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape

    if np.isscalar(scores):
        if scores < thr:
            return imgcv
        cls_indx = int(cls_inds)
        thick = int((h + w) / 1000)
        cv2.rectangle(imgcv,
                      (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])),
                      (int(colors[cls_indx][0]), int(colors[cls_indx][1]), int(colors[cls_indx][2])), 
                      thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores)
        cv2.putText(imgcv, mess, (int(bboxes[0]), int(bboxes[1]) - 7),
                    0, 1e-3 * h, (int(colors[cls_indx][0]), int(colors[cls_indx][1]), int(colors[cls_indx][2])), 
                    thick)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)

    else:
        for i, box in enumerate(bboxes):
            if scores[i] < thr:
                continue
            cls_indx = int(cls_inds[i])
            box = [int(_) for _ in box]
            thick = int((h + w) / 1000)
            cv2.rectangle(imgcv,
                          (box[0], box[1]), (box[2], box[3]),
                          colors[cls_indx], thick)
            mess = '%s: %.3f' % (labels[cls_indx], scores[i])
            cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                        0, 1e-3 * h, colors[cls_indx], thick)
            if fps >= 0:
                cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)
    return imgcv

def draw_region(im, regions, divide_region):
    base = int(np.ceil(pow(len(divide_region), 1. / 3)))
    colors = [_to_color(x, base) for x in range(len(divide_region))]
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    n=0
    region_count=1
    for i, task in enumerate(regions):
        if region_count > divide_region[n]: 
            region_count=1
            n+=1 
        box = list(task[0])
        thick = int((h + w) / 1000)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[n], thick)
        region_count+=1
        
    return imgcv

im_path = args.directory
cam = args.cam
if cam >= 0:
    capture = cv2.VideoCapture(cam)
im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
im_iter = iter(im_fnames)
start_distance = 800
count = 1

if args.record:
    video_resolution = (1920,1080)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter('./output.avi', fourcc, 20.0, video_resolution)

if args.log:
    file_txt = "/media/student/HDD 1/Marcus/M2Det-Guns/"+args.directory+"detections.txt"

    if os.path.exists(file_txt):
       os.remove(file_txt)

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """ 
    tree = ET.parse(filename)

    for obj in tree.findall('object'):
        obj.find('name').text = 'pistol'
        bbox = obj.find('bndbox')

        return torch.Tensor([float(bbox.find('xmin').text), \
            float(bbox.find('ymin').text), \
            float(bbox.find('xmax').text), \
            float(bbox.find('ymax').text)]).unsqueeze(0)

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].expand(A, B, 2),
                       box_b[:, 2:].expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].expand(A, B, 2),
                       box_b[:, :2].expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union

top_score_buffer = []
while True:
    if cam < 0:
        try:
            fname = next(im_iter)
        except StopIteration:
            break
        if 'm2det' in fname: continue # ignore the detected images
        image = cv2.imread(fname, cv2.IMREAD_COLOR)

    else:
        ret, image = capture.read()
        if not ret:
            cv2.destroyAllWindows()
            capture.release()
            video_out.release()
            break
    loop_start = time.time()
    w,h = image.shape[1],image.shape[0]
    img = _preprocess(image).unsqueeze(0)

    if cfg.test_cfg.cuda:
        img = img.cuda()
    scale = torch.Tensor([w,h,w,h])
    out = net(img)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0]*scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    allboxes = []
    for j in range(1, cfg.model.m2det_config.num_classes):
        inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = keep[:cfg.test_cfg.keep_per_class]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist()+[j] for _ in c_dets])

    loop_time = time.time() - loop_start
    allboxes = np.array(allboxes)

    if args.log:
        file = open(args.directory+"detections.txt","a") 

        anno_path = args.directory+'anno/'+fname.split(args.directory)[-1].replace('.jpg','.xml')
        
        if os.path.exists(anno_path):
            gt_box = parse_rec(anno_path)

            top_intersect_index = []
            top_intersect_index_ad = []
            btop_score_ad = False

            if len(allboxes) != 0:
                boxes = torch.Tensor(allboxes[:,:4])
                inter = jaccard(gt_box,boxes).cpu().numpy().squeeze(0)
                top_intersect_index = np.where(inter >= 0.3) #STATIC
                boxes = (allboxes[:,:4])[top_intersect_index]

                if len(inter[top_intersect_index] != 0):
                    top_score_index = np.argmax((allboxes[:,4])[top_intersect_index])
                    top_score = (allboxes[:,4][top_intersect_index])[top_score_index]
                    top_score_box = boxes[top_score_index]
                    score_thr = 0.2 #STATIC

                    if args.ad_nms and top_score < score_thr: 
                        w,h = image.shape[1],image.shape[0]

                        # Divide into regions
                        overlap = 0.1
                        # divide_region = [12,6,3]
                        bbox_w = top_score_box[2]-top_score_box[0]
                        bbox_h = top_score_box[3]-top_score_box[1]

                        if top_score_box[3] <= 0.25*h:
                            if bbox_w >= bbox_h:
                                region_size = int(w/(bbox_w + top_score_box[3]))

                            elif bbox_w < bbox_h:
                                region_size = int(w/(bbox_h + top_score_box[3]))

                        elif top_score_box[3] <= 0.5*h:
                            if bbox_w >= bbox_h:
                                region_size = int(w/(bbox_w/2 + top_score_box[3]/2))

                            elif bbox_w < bbox_h:
                                region_size = int(w/(bbox_h/2 + top_score_box[3]/2))

                        elif top_score_box[3] > 0.5*h:
                            if bbox_w >= bbox_h:
                                region_size = int(w/(bbox_w/3 + top_score_box[3]/3))

                            elif bbox_w < bbox_h:
                                region_size = int(w/(bbox_h/3 + top_score_box[3]/3))

                        divide_region = [region_size, int(region_size/1.5), int(region_size/3)]
                        region_list = divideImage((w, h), divide_region, overlap_rate=overlap)
                        task_list = createObjectDectionTasks(image, region_list)        

                        allboxes_ad = []

                        for task in task_list:
                            task_w, task_h = task[1].shape[1], task[1].shape[0]
                            temp_boxes=[]

                            if (top_score_box[0] >= task[0][0] and top_score_box[2] <= task[0][2] and
                                top_score_box[1] >= task[0][1] and top_score_box[3] <= task[0][3]):
                                print(task[0])
                                img = _preprocess(task[1]).unsqueeze(0)
                                if cfg.test_cfg.cuda:
                                    img = img.cuda()
                                scale = torch.Tensor([task_w,task_h,task_w,task_h])
                                out = net(img)
                                boxes, scores = detector.forward(out, priors)
                                boxes = (boxes[0]*scale).cpu().numpy()
                                scores = scores[0].cpu().numpy()

                                for j in range(1, cfg.model.m2det_config.num_classes):
                                    inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
                                    if len(inds) == 0:
                                        continue
                                    c_bboxes = boxes[inds]
                                    c_scores = scores[inds, j]
                                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                                    soft_nms = cfg.test_cfg.soft_nms
                                    keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
                                    keep = keep[:1]
                                    c_dets = c_dets[keep, :]
                                    temp_boxes.extend([_.tolist()+[j] for _ in c_dets])
                                    temp_boxes = np.array(temp_boxes).astype(np.float32, copy=False)
                                    # im2show = draw_detection(task[1], temp_boxes[:,:4], temp_boxes[:,4], temp_boxes[:,5], -1, thr=0.1)
                                    # cv2.imshow('test', im2show)                       
                                    # cv2.waitKey(2000)
                                    # cv2.imwrite(args.directory+'outputs/'+'m2det_ADNMS_{}'.format(fname.split(args.directory)[1]), im2show)
                                    temp_boxes[:,:4] = temp_boxes[:,:4]+np.array([task[0][0], task[0][1], task[0][0],task[0][1]])
                                    temp_boxes = temp_boxes.tolist()
                                    allboxes_ad.extend(temp_boxes)
                            
                        allboxes_ad = np.array(allboxes_ad).astype(np.float32, copy=False)
                        # keep = nms(allboxes_ad, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
                        # keep = keep[:cfg.test_cfg.keep_per_class]
                        # allboxes_ad = allboxes_ad[keep, :]

                        if len(allboxes_ad) != 0:
                            boxes = torch.Tensor(allboxes_ad[:,:4])
                            inter = jaccard(gt_box,boxes).cpu().numpy().squeeze(0)
                            top_intersect_index_ad = np.where(inter >= 0.2) #STATIC 
                            boxes = boxes[top_intersect_index_ad]
                            print(boxes)

                            if len(inter[top_intersect_index_ad] != 0):
                                top_score_index_ad = np.argmax((allboxes_ad[:,4])[top_intersect_index_ad])
                                top_score_ad = (allboxes_ad[:,4][top_intersect_index_ad])[top_score_index_ad]
                                boxes = boxes[top_score_index_ad,:4].cpu().numpy()
                                if top_score_ad > top_score: 
                                    btop_score_ad = True
                                    allboxes_ad = [boxes[0], boxes[1],
                                                   boxes[2], boxes[3],
                                                   top_score_ad, 1 
                                                  ]
                                    allboxes_ad = np.array(allboxes_ad)
                                    print("before "+str(top_score))
                                    top_score = top_score_ad
                                    print("improve "+str(top_score_ad))
     
                    file.write(str(top_score)+'\n')
                    # top_score_buffer.append(top_score)

                else:
                    file.write('0'+'\n')
                    # top_score_buffer.append(0)

            else:   
                file.write('0'+'\n')
                # top_score_buffer.append(0)
            
            # if count % 3 == 0: 
            #     file.write(str(np.max(top_score_buffer))+" "+str(start_distance)+'\n')
            #     top_score_buffer = []
            #     start_distance = start_distance - 15.68627451
            # count+=1

        file.close


    if (len(allboxes) != 0):
        boxes = allboxes[:,:4]
        scores = allboxes[:,4]
        cls_inds = allboxes[:,5]

        if args.ad_nms and btop_score_ad and "top_intersect_index_ad" in locals():
                boxes = allboxes_ad[:4]
                scores = allboxes_ad[4]
                cls_inds = allboxes_ad[5]

        # print('\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
        #         ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)]))
        fps = 1.0 / float(loop_time) if cam >= 0 else -1 
        im2show = draw_detection(image, boxes, scores, cls_inds, fps, thr=0.1)
        # im2show = draw_region(image, task_list, divide_region)
    else:
        im2show=image

    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))

    if args.show:
        cv2.imshow('test', im2show)
        if cam < 0:
            cv2.waitKey(5000)
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                capture.release()
                video_out.release()
                break

    if cam < 0:
        print(fname.split(args.directory)[1])
        cv2.imwrite(args.directory+'outputs/'+'m2det_{}'.format(fname.split(args.directory)[1]), im2show)

    if args.record:
        #frame = cv2.flip(im2show,0)
        im2show = cv2.resize(im2show,video_resolution)
        video_out.write(im2show)


