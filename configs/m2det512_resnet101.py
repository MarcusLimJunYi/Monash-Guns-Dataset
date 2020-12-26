model = dict(
    type = 'm2det',
    input_size = 512,
    init_net = True,
    pretrained = 'None',
    m2det_config = dict(
        backbone = 'resnet101',
        net_family = 'res', # vgg includes ['vgg16','vgg19'], res includes ['resnetxxx','resnextxxx']
        base_out = [1,2,4],
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = False,
        smooth = True,
        num_classes = 2,
        ),
    rgb_means = (104, 117, 123),
    p = 0.6,
    anchor_config = dict(
        step_pattern = [8, 16, 32, 64, 128, 256],
        size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        ),
    save_eposhs = 1,
    weights_save = 'weights/'
    )

train_cfg = dict(
    cuda = True,
    warmup = 2,
    per_batch_size = 2,
    lr_sgd = [0.0004, 0.0002, 0.00004, 0.000004, 0.0000004],
    lr_adam = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        COCO = [150, 200, 230, 260, 300],
        VOC = [100, 150, 200, 250, 300], # unsolve
        MGD = [90, 110, 130, 150, 300], 
        ),
    print_epochs = 10,
    num_workers= 8,
    )

test_cfg = dict(
    cuda = True,
    topk = 0,
    iou = 0.45,
    soft_nms = True,
    score_threshold = 0.1,
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

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC = dict(
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')],
        test_sets = [('2007', 'trainval')],
        eval_sets = [('2007', 'test')],
        ),

    COCO = dict(
        train_sets = [('2017', 'train'), ('2017', 'val')],
        eval_sets = [('2017', 'val')],
        test_sets = [('2015', 'test-dev')],
        ),

    MGD = dict(
        train_sets = [('2020', 'train')],
        test_sets = [('2020', 'trainval')],
        eval_sets = [('2020', 'test')],
        ),

    )

VOCroot = "dataset/VOCdevkit/"
COCOroot = "dataset/coco2017/"
MGDroot = "dataset/MGD/"

