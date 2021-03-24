default_net_params = dict(
    backcast_length = 1200,
    forecast_length = 200,
    batch_size = 32,
    classes = ['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC'],
    backcast_multiplier = 6,
    hidden_layer_units=16,
    nb_blocks_per_stack=3,
    thetas_dim=[7,8]
)


old_exp_net_params = dict(
    backcast_length = 1200,
    forecast_length = 200,
    batch_size = 64,
    classes = ['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC'],
    backcast_multiplier = 6,
    hidden_layer_units=16,
    nb_blocks_per_stack= 8,
    thetas_dim=[16,32]
)

old2_exp_net_params = dict(
    backcast_length = 1200,
    forecast_length = 200,
    batch_size = 128,
    classes = ['LBBB', 'Normal', 'RBBB', 'AF', 'STE', 'PAC', 'PVC'],
    backcast_multiplier = 6,
    hidden_layer_units=64,
    nb_blocks_per_stack= 8,
    thetas_dim=[32,64]
)

exp_net_params = dict(
    backcast_length = 500,
    forecast_length = 100,
    batch_size = 16,
    classes =  ['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC'],
    backcast_multiplier = 7,
    hidden_layer_units=16,
    nb_blocks_per_stack= 4,
    thetas_dim=[7, 16]
)



epoch_limit = 25

criterion="MSE" #"HUBER" / "MSLE" / "LOGCOSH" / "MSE" / "FASTDTW"

leads_dict_available= False
# leads are provided REAL - 1
leads_dict = {
    'AF': 2,
    'I-AVB':1,
    'LBBB' : 11,
    'Normal':2,
    'PAC':1,
    'PVC':1,
    'RBBB':0,
    'STD':2,
    'STE':4  
}
