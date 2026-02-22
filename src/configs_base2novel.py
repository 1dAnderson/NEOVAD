
def build_config(dataset):
    cfg = type('', (), {})()
        # base settings
    cfg.feat_dim = 512  
    # cfg.hid_dim = 128
    cfg.dropout_gat = 0.6
    cfg.out_dim = 32
    cfg.alpha = 0.1
    cfg.train_bs = 128
    cfg.workers = 4
    cfg.prefix = 16
    cfg.postfix = 16
    cfg.device = "cuda:0"
    cfg.load_ckpt = False
    cfg.WANDB = False
    cfg.eval_on_cpu = False
    cfg.fixed_prompt = True
    cfg.temporal = True
    cfg.head_num = 4
    # loss setting
    cfg.lamda1 = 1
    cfg.lamda2 = 1 #1
    cfg.lamda3 = 2 #2
    if dataset in ['ucf', 'ucf-crime']:
        cfg.dataset = 'ucf-crime'
        cfg.model_name = 'ucf_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '/data/dengyunhui/datasets/UCF-Crime/clip/UCF-CLIP-TSA/'
        # ov-vad training
        cfg.train_list = 'list/base2novel/ucf-vit-train_base+normal.list'
        # w-vad training
        # cfg.train_list = 'list/ucf/ucf-vit-train.list'

        cfg.test_list = 'list/ucf/ucf-vit-test.list'
        cfg.gt = 'list/ucf/gt-ucf-vit.npy'
        cfg.token_feat = 'list/ucf/ucf-prompt_1_not16_859.npy'
        # cfg.token_feat = 'list/ucf/ucf-prompt_gpt35_2.npy'
        cfg.clslist = "./list/prompt/ucf_cls.txt"
        cfg.has_feature_input = True
        # base2novel test: only base
        # cfg.test_list = './list/base2novel/ucf-vit-test_base+normal.list'
        # cfg.gt = './list/base2novel/gt-ucf_base.npy'
        # base2novel test: only novel
        # cfg.test_list = './list/base2novel/ucf-vit-test_novel+normal.list'
        # cfg.gt = './list/base2novel/gt-ucf_novel.npy'
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # training settings
        cfg.temp = 0.09
        cfg.seed = 2 #ov-vad:2; w-vad:13 ⭐
        #Pseudo label loss settings
        # cfg.pse_alpha = 0.25
        # cfg.pse_gamma = 2.0
        # cfg.pse_threshold = 0.4

        # focal loss nums
        cfg.focal_nums = 14

        # text adapt setting 
        cfg.text_adapt_until = 4 #4
        cfg.text_adapt_weight = 0.1

        # test settings
        cfg.test_bs = 1
        cfg.ckpt_path = 'ucf__ckpt_best_tmp.pkl'
        # prompt
        cfg.preprompt = False
        cfg.backbone = 'ViT-B/16'
        cfg.mask_rate = 0.55
        cfg.save_dir = './ckpt/base2novel_ucf/'
        cfg.logs_dir = 'log_temp.log'
        cfg.max_epoch = 30
        cfg.max_seqlen = 200 #200
        cfg.lr = 1e-3 #1e-3
        cfg.std_init = 0.01
        cfg.head_num = 4 #4
        cfg.cls_hidden = 128
    elif dataset in ['sh', 'shanghaitech']:
        cfg.dataset = 'shanghaitech'
        cfg.model_name = 'sh_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '/data/dengyunhui/all_datasets/shanghaiTech/sh/features/'
        # ov-vad training
        # cfg.train_list = 'list/base2novel_sh/half/sh-vit-train_base+normal.list'
        # w-vad training
        cfg.train_list = 'list/sh/sh-vit-train_flag.list'
        cfg.test_list = 'list/sh/sh-vit-test_flag.list'
        cfg.gt = 'list/sh/gt-sh.npy'
        cfg.clslist = "./list/prompt/class_sh.txt"
        # cfg.test_list = 'list/base2novel_sh/half/sh-vit-test_flag-single.list'
        # cfg.gt = 'list/base2novel_sh/half/gt-sh_flag-single.npy'
        cfg.token_feat = 'list/sh/sh-prompt_12.npy'
        # cfg.clslist = "./list/prompt/sh_cls_re2.txt"
        # #fully
        # cfg.test_list = 'list/sh/sh-vit-test_flag.list'
        # cfg.gt = 'list/sh/gt-sh.npy'
        # cfg.token_feat = 'list/sh/sh-prompt_all.npy'
        # cfg.clslist = "./list/prompt/class_sh.txt"
        cfg.has_feature_input = True
        # base2novel test
        # cfg.test_list = 'list/base2novel_sh/half/sh-vit-test_novel+normal.list'
        # cfg.gt = 'list/base2novel_sh/half/gt-sh_novel+normal.npy'

        # cfg.test_list = 'list/base2novel_sh/half/sh-vit-test_base+normal.list'
        # cfg.gt = 'list/base2novel_sh/half/gt-sh_base+normal.npy'
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # training settings
        cfg.temp = 0.09
        cfg.seed = 1 #ov-vad:1;  ⭐

        # focal loss nums
        cfg.focal_nums = 18 # ov-vad: 12; w-vad: 18

        # text adapt setting 
        cfg.text_adapt_until = 4 #2
        cfg.text_adapt_weight = 0.1

        # test settings
        cfg.test_bs = 1
        cfg.ckpt_path = 'sh__ckpt_best_tmp.pkl'
        # prompt
        cfg.preprompt = False
        cfg.backbone = 'ViT-B/16'
        cfg.mask_rate = 0.9
        cfg.save_dir = './ckpt/base2novel_sh/'
        cfg.logs_dir = 'log_temp.log'
        cfg.max_epoch = 60
        cfg.max_seqlen = 120
        cfg.lr = 5e-5 #5e-4 
        cfg.head_num = 4 #4
        cfg.std_init = 0.02
        cfg.cls_hidden = 128
    elif dataset in ['ub', 'ubnormal']:
        cfg.dataset = 'ubnormal'
        cfg.model_name = 'ub_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '/data/dengyunhui/all_datasets/ubnormal/UBnormal_features_reorg/'
        cfg.train_list = 'list/ubnormal/ub-vit-train_or.list'
        cfg.test_list = 'list/ubnormal/ub-vit-test.list'
        cfg.gt = 'list/ubnormal/gt.npy'
        cfg.token_feat = 'list/ubnormal/ubnormal-prompt_1_not16.npy'
        cfg.name2clspath = "list/ubnormal/name2cls.json" 
        cfg.clslist = "./list/prompt/class_ubnormal.txt"
        cfg.has_feature_input = True
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # training settings
        cfg.temp = 0.09
        cfg.seed = 1 #1 ⭐
        # test settings
        cfg.test_bs = 1
        cfg.ckpt_path = 'ub__ckpt_best_tmp.pkl'
        # cfg.ckpt_path = ''
        # prompt

        # text adapt setting 
        cfg.text_adapt_until = 4 #4
        cfg.text_adapt_weight = 0.1

        # focal loss nums
        cfg.focal_nums = 23 # 23

        cfg.preprompt = False
        cfg.backbone = 'ViT-B/16'
        cfg.mask_rate = 0.9
        cfg.save_dir = './ckpt/base2novel_ub/'
        cfg.logs_dir = 'log_temp.log'
        cfg.max_epoch = 200
        cfg.max_seqlen = 450 #450
        cfg.lr = 1e-3 #1e-3
        cfg.std_init = 0.01
        cfg.head_num = 4 #4  
        cfg.cls_hidden = 128
    elif dataset in ['xd', 'xd-violence']:
        cfg.dataset = 'xd-violence'
        cfg.model_name = 'xd_'
        cfg.metrics = 'AP'
        cfg.feat_prefix = '/data/dengyunhui/all_datasets/XD-Violence/xd/features/'
        # ov-vad training
        cfg.train_list = 'list/xd_single/base2novel/xd-vit-train_base+normal.list'
        # w-vad training
        # cfg.train_list = 'list/xd_single/xd-vit-train.list'

        # cfg.test_list = 'list/xd_single/base2novel/xd-vit-test_all.list'
        # cfg.gt = 'list/xd_single/base2novel/gt-xd_vit_all.npy'

        cfg.token_feat = 'list/ucf/ucf-prompt_1_not16_859.npy'

        cfg.clslist = "list/prompt/xd_cls.txt"
        cfg.cls2flag = "list/xd_single/base2novel/clsinfo/xd_label2cls.json"
        cfg.has_feature_input = True
        # base2novel test: only base
        # cfg.test_list = 'list/xd_single/base2novel/xd-vit-test_base.list'
        # cfg.gt = 'list/xd_single/base2novel/gt-xd_vit_base.npy'
        # cfg.gt = 'list/xd_single/base2novel/gt-xd_vit_base.npy'
        # base2novel test: only novel
        cfg.test_list = './list/xd_single/base2novel/xd-vit-test_novel+normal.list'
        cfg.gt = './list/xd_single/base2novel/gt-xd_vit_novel+normal.npy'
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # training settings
        cfg.temp = 0.09
        cfg.seed = 20 #ov-vad:20;  ⭐
        #Pseudo label loss settings
        # cfg.pse_alpha = 0.25
        # cfg.pse_gamma = 2.0
        # cfg.pse_threshold = 0.4

        # focal loss nums
        cfg.focal_nums = 7

        # text adapt setting 
        cfg.text_adapt_until = 8 #8
        cfg.text_adapt_weight = 0.1

        cfg.lamda2 = 0.6

        # test settings
        cfg.test_bs = 1
        cfg.ckpt_path = 'xd__ckpt_best_tmp.pkl'
        # prompt
        cfg.preprompt = False
        cfg.backbone = 'ViT-B/16'
        cfg.mask_rate = 0.9
        cfg.save_dir = './ckpt/base2novel_xd/'
        cfg.logs_dir = 'log_temp.log'
        cfg.max_epoch = 60
        cfg.max_seqlen = 200 #200
        cfg.lr = 7e-4 #7e-4
        cfg.std_init = 0.01
        cfg.head_num = 4 #4
        cfg.cls_hidden = 128
        
    return cfg
