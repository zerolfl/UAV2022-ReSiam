from parameter.params import TrackerParams

def parameters():
    params = TrackerParams()

    # These are usually set from outside
    params.debug = 0                        # Debug level
    params.visualization = False            # Do visualization

    # Use GPU or not (IoUNet requires this to be True)
    params.use_gpu = True

    # Patch sampling parameters
    params.context_amount = 0.5
    params.exemplar_side = 127
    params.instance_side = 255
    
    # Pos and size limitations
    params.clamp_target_sz = True
    params.clamp_position = True
    
    ########### ReCF ###########
    # ReCF parameters
    params.learning_rate = 1            # learning rate
    params.gamma_H = 28                 # Parameter on historical response regularizatio
    params.gamma_I = 102.2              # Parameter on Inferred response regularization
    # ReCF ADMM parameters
    params.admm_iterations = 3          # Iterations
    params.mu = 100                     # Initial penalty factor
    params.beta = 500                   # Scale step
    params.mu_max = 100000              # Maximum penalty factor
    # Other parameters
    params.output_sigma_factor = 1 / 16.
    params.reg_window_max = 1e5
    params.reg_window_min = 1e-3
    # ReCF pos detection params
    params.use_resp_newton = True        # Refine the resulting position using Newton's method
    params.newton_iterations = 5
    params.use_detection_sample = False  # Use the sample extracted at the detection stage for learning
    # Training parameters
    params.train_skipping = 1            # How often to run training of online model (every n-th frame)
    
    ########### RPN Head ###########
    # Anchors setting
    params.anchor_stride = 8
    params.anchor_ratios = [0.33, 0.5, 1, 2, 3]
    params.anchor_scales = [8]
    # Other parameters
    params.penalty_factor = 0.1
    params.penalty_cos_window_factor = 0.4
    params.smooth_size_lr = 0.42
    
    ############ Advanced state analysis ############
    # Only supports for RPNHead
    params.enlarge_search_mode = True
    params.enlarge_search_side = 433
    params.target_not_found_threshold = 0.5  # Threshold to detect target missing (usually >penalty_cos_window_factor)

    ############ Features ############
    params.feat = 'd'  # h: fhog, gray, cn; d: alexnet-l5, hd: all
    params.use_alexnet_pad = True  # True False
    params.net_path = 'resiam.pth'
    
    params.use_recf = True  # Op. True False
    params.use_rpn_head = True  # Op.
    
    return params
