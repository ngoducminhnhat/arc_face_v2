class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 1200
    metric = 'arc_margin'
    easy_margin = False
    use_se = True
    loss = 'focal_loss'

    display = False
    finetune = False


    train_root = '/kaggle/input/arcfae/dataset/not_mask'
    train_list = '/kaggle/input/txt-file/txt/not_mask_8.txt'
    val_list = '/kaggle/input/txt-file/txt/not_mask_2.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/kaggle/input/arcfae/dataset/lfw-align-128'
    lfw_test_list = '/kaggle/input/txt-file/txt/lfw_test_pair.txt'

    # checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = '/kaggle/input/resnet-pth12/resnet18_110.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
