## This is the first initial network config. 
## Aims to be as simple as possible. 

metadata = {
   'run_name': 'v0: vanilla PPO' 
}

train_settings = {
    'gamma': 1.0,
    'update_opponent_wr': 0.55,
    'num_steps': 1_000_000, # steps to update the network
    'batch_size': 2000
}


MCTS_settings = {
    'n_states_sample': 800 # MCTS evaluations per move
}

architecture_settings = {
    'n_residual_blocks': 4,
    'block_width': 1000
}





