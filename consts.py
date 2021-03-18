from best_metrics import Best


class GlobalConsts:
    single_gpu = True
    load_model = False
    save_grad = False
    dataset = "mosi"
    data_path = "/workspace/dataset/"
    log_path = None
    padding_len = -1
    include_zero = True
    # cellDim = 150
    # normDim = 100
    # hiddenDim = 300
    config = {
      "seed": 0,
      "batch_size": 2,
      "epoch_num": 50,
      "cuda": 0,
        
      "global_lr": 1e-4,
      "gru_lr": 1e-4, 
      "beta1": 0.9,
      "beta2": 0.999,
      "eps": 1e-8,
      'weight_decay': 1e-2,
      'momentum': 0.9,
    
      "gnn_dropout": 0.1,
      "num_modality": 3,
      "num_frames": 50,
      "temporal_connectivity_order": 5,
      "num_vision_aggr": 1,
      "num_text_aggr": 1,
      "num_audio_aggr": 1,
      "text_dim": 300,
      "audio_dim": 5,
      "vision_dim": 20,
      "graph_conv_in_dim": 512,
      "graph_conv_out_dim": 512,
      "gat_conv_num_heads": 4,
        
      "transformer_nhead": 4,
      "transformer_nhid": 1024,
      "transformer_nlayers": 6,
    }


    device = None

    best = Best()

    def logParameters(self):
        print( "Hyperparameters:")
        for name in dir(GlobalConsts):
            if name.find("__") == -1 and name.find("max") == -1 and name.find("min") == -1:
                print( "\t%s: %s" % (name, str(getattr(GlobalConsts, name))))
