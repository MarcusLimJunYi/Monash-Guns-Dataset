model = dict(
    type = 'm2det',
    input_size = 512,
    init_net = True,
    pretrained = 'weights/vgg16_reducedfc.pth',
    m2det_config = dict(
        backbone = 'vgg16',
        net_family = 'vgg', # vgg includes ['vgg16','vgg19'], res includes ['resnetxxx','resnextxxx']
        # base_out = [22,34],
        base_out = [15,22,34], # [22,34] for vgg, [2,4] or [3,4] for res families
        # backbone = 'resnet101',
        # net_family = 'res', # vgg includes ['vgg16','vgg19'], res includes ['resnetxxx','resnextxxx']
        # base_out = [2,4],
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = True,
        smooth = True,
        num_classes = 2,
        ),
    # rgb_means = (104, 117, 123),
    rgb_means = (133, 127, 117), #RGB MEAN V1
    p = 0.6,
    anchor_config = dict(
        step_pattern = [8, 16, 32, 64, 128, 256],
        size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            #[30.72, 76.8, 168.96, 261.12, 353.28, 445.44]
            #[76.8, 168.96, 261.12, 353.28, 445.44, 537.6]
        # size_pattern = [0.03, 0.08, 0.23, 0.37, 0.58, 0.71, 0.91], #ANCHOR V1
        # size_pattern = [0.0254, 0.0762, 0.2266, 0.3691, 0.5801, 0.7051, 0.9141], #ANCHOR V2
            # [13.0, 39.01, 116.02, 188.98, 297.01, 361.01]
            # [39.01, 116.02, 188.98, 297.01, 361.01, 468.02]
        #size_pattern = [0.02929, 0.08693, 0.23437, 0.48828, 0.63085, 0.80468, 0.92382], #ANCHOR V3
        ),
    # 15,29, 44,51, 85,111, 136,257, 174,120, 250,373, 299,211, 412,323, 473,464  
    # 15, 44.51, 120, 250, 323, 412, 473  
    save_eposhs = 10,
    weights_save = 'weights/'
    )

train_cfg = dict(
    cuda = True,
    warmup = 5,
    per_batch_size = 13,
    lr = [0.001, 0.0004, 0.0002, 0.00004, 0.000004],
    #lr = [0.000004, 0.000002, 0.0000004, 0.00000004, 0.000000004],
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        COCO = [90, 110, 130, 150, 300],
        #COCO = [150, 200, 230, 260, 300],
        VOC = [100, 150, 200, 250, 300], # unsolve
        GUN = [100, 150, 200, 250, 300], # unsolve
        ),
    print_epochs = 10,
    num_workers= 8,
    )

test_cfg = dict(
    cuda = True,
    topk = 0,
    iou = 0.45,
    soft_nms = True,
    score_threshold = 0.5,
    keep_per_class = 50,
    save_folder = 'eval'
    )

loss = dict(overlap_thresh = 0.5,
            prior_for_matching = True,
            bkg_label = 0,
            neg_mining = True,
            neg_pos = 3,
            neg_overlap = 0.5,
            encode_target = False)

optimizer = dict(type='Adam', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC = dict(
        #train_sets = [('2007', 'trainval'), ('2012', 'trainval')],
        train_sets = [('2007', 'train')],
        test_sets = [('2007', 'trainval')],
        eval_sets = [('2007', 'test')],
        ),

    COCO = dict(
        train_sets = [('2014', 'train'), ('2014', 'valminusminival')],
        eval_sets = [('2014', 'minival')],
        test_sets = [('2015', 'test-dev')],
        ),

    GUN = dict(
        train_sets = [('_Recorded', 'train')],
        eval_sets = [('_Recorded', 'test')],
        ),

    )

import os
home = os.path.expanduser("~")
# VOCroot = os.path.join(home,"data/VOCdevkit/")
# VOCroot = "/media/student/HDD 1/data/VOCdevkit2007_512x512/"
#VOCroot = "/media/student/HDD 1/data/Xiaomi_Xiaofang_512x512/"
#VOCroot = "/media/student/HDD 1/data/VOC_Xiaomi_Combine_512x512/"
VOCroot = "/media/student/HDD 1/data/VOC_Xiaomi_Combine_512x512_V2/"
# VOCroot = "/media/student/HDD 1/data/VOC_Xiaomi_Combine_512x512_V2_Rescale/"
# VOCroot="/media/student/HDD 1/data/Xiaomi_V2_512x512"
#VOCroot = "/media/student/HDD 1/data/Recorded_Videos/"
#VOCroot = "/media/student/HDD 1/data/UCFCrime_Test/"
#COCOroot = os.path.join(home,"data/coco/")
#COCOroot = "/media/student/HDD 1/data/Xiaomi_Xiaofang_512x512/"
COCOroot = "/media/student/HDD 1/data/UCFCrime_Test_Good/"
# COCOroot = "/media/student/HDD 1/data/EAAI_Test/"
# COCOroot = "/media/student/HDD 1/data/Granada_Test/"
# COCOroot = "/media/student/HDD 1/data/VOC_Xiaomi_Combine_512x512_V2/"
GUNroot= "/media/student/HDD 1/data/GUN_Recorded/"


