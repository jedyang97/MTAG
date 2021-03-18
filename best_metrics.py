class Best():
    best_epoch = -1
    best_test_epoch = -1
    best_val_epoch = -1
    best_val_epoch_lr = -1
    best_test_epoch_lr = -1

    test_mae_at_valid_best = -1
    test_cor_at_valid_best = -1
    test_acc_at_valid_best = -1
    test_ex_zero_acc_at_valid_best = -1
    test_acc_7_at_valid_best = -1
    test_f1_raven_at_valid_best = -1
    test_f1_mfn_at_valid_best = -1
    test_f1_mult_at_valid_best = -1

    valid_mae_at_test_best = -1
    valid_cor_at_test_best = -1
    valid_acc_at_test_best = -1
    valid_ex_zero_acc_at_test_best = -1
    valid_acc_7_at_test_best = -1
    valid_f1_raven_at_test_best = -1
    valid_f1_mfn_at_test_best = -1
    valid_f1_mult_at_test_best = -1

    checkpoints_val_mae = {}
    checkpoints_test_mae = {}
    checkpoints_val_ex_0_acc = {}
    checkpoints_test_ex_0_acc = {}

    max_train_f1 = -1
    max_test_f1 = -1
    max_valid_f1 = -1

    max_valid_f1_mfn = -1
    max_valid_f1_raven = -1
    max_valid_f1_mult = -1

    max_test_f1_mfn = -1
    max_test_f1_raven = -1
    max_test_f1_mult = -1

    max_test_prec = -1
    max_valid_prec = -1
    max_train_prec = -1
    max_train_recall = -1
    max_test_recall = -1
    max_valid_recall = -1
    max_train_acc = -1
    max_valid_acc = -1
    max_test_acc = -1
    max_valid_ex_zero_acc = -1
    max_test_ex_zero_acc = -1
    max_valid_acc_5 = -1
    max_test_acc_5 = -1
    max_valid_acc_7 = -1
    max_test_acc_7 = -1

    test_cor_at_valid_max = -1
    test_acc_at_valid_max = -1
    test_ex_zero_acc_at_valid_max = -1
    test_acc_5_at_valid_max = -1
    test_acc_7_at_valid_max = -1
    test_f1_at_valid_max = -1

    test_f1_mfn_at_valid_max = -1
    test_f1_raven_at_valid_max = -1
    test_f1_mult_at_valid_max = -1

    test_prec_at_valid_max = -1
    test_recall_at_valid_max = -1

    min_train_mae = 9999
    min_test_mae = 9999
    max_test_cor = -1
    min_valid_mae = 9999
    max_valid_cor = -1
    test_mae_at_valid_min = 9999
    test_cor_at_valid_max = -1

    iemocap_emos = ["Neutral", "Happy", "Sad", "Angry"]
    split = ['train', 'valid', 'test_at_valid_max', 'test']
    max_iemocap_f1 = {}
    max_iemocap_acc = {}
    best_iemocap_epoch = {}
    for sp in split:
        max_iemocap_f1[sp] = {}
        max_iemocap_acc[sp] = {}
        best_iemocap_epoch[sp] = {}
        for em in iemocap_emos:
            max_iemocap_f1[sp][em] = -1
            max_iemocap_acc[sp][em] = -1
            best_iemocap_epoch[sp][em] = -1

    pom_cls = ["Confidence", "Passionate", "Voice pleasant", "Dominant", "Credible", "Vivid", "Expertise",
               "Entertaining", "Reserved", "Trusting", "Relaxed", "Outgoing", "Thorough",
               "Nervous", "Sentiment", "Persuasive", "Humorous"]

    max_pom_metrics = {metric: {} for metric in ['acc', 'corr']}
    for metric in ['acc', 'corr']:
        for sp in split:
            max_pom_metrics[metric][sp] = {}
            for cls in pom_cls:
                max_pom_metrics[metric][sp][cls] = -1

    best_pom_mae = {}
    for sp in split:
        best_pom_mae[sp] = {}
        for cls in pom_cls:
            best_pom_mae[sp][cls] = 9999
