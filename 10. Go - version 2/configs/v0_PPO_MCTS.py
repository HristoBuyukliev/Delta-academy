## This is the first initial network config. 
## Aims to be as simple as possible. 

metadata = {
   'run_name': 'v0: vanilla PPO' 
}

train_settings = {
    'gamma': 1.0,
    'update_opponent_wr': 0.55,
    'num_steps': 1_000_000, # steps to update the network
    'batch_size': 2000,
    'pcr': 0.25, # playout cap randomization: https://arxiv.org/pdf/1902.10565.pdf, section 3.1
    'full_search': 200,
    'small_search': 50,
    'c_puct': 2, # 5 for alphazero, 1.1 for kataGo
    'algo_tree': 'PUCT', # one of {UCT, PUCT}
    'root_dirichlet_noise': 0.25,
    'root_alpha': 0.03
}


MCTS_settings = {
    'n_states_sample': 800 # MCTS evaluations per move
}

architecture_settings = {
    'n_residual_blocks': 3,
    'block_width': 64, 
    'global_pooling': True,
    'transformer_blocks': 0
}





