your_project/
├── dataset_configs/          # Your map config YAMLs
│   └── 10-medium-mazes/
│       └── maps.yaml....
├── prepared_agent_centric_dataset_v3/ # Output of preprocessing
│   ├── train/                  # Training NPZ files here
│   │   └── map_seed_0000_agents8.npz
│   │   └── ...
│   └── test/                   # Testing NPZ files here
│       └── map_seed_1000_agents16.npz
│       └── ...
├── training_output/            # Output of training
│   ├── logs/                   # TensorBoard logs
│   ├── checkpoints/            # Intermediate checkpoints
│   └── best_model.pth          # Best model checkpoint
├── create_env.py             # Your custom environment setup (if any)
├── preprocess_data.py        # Your corrected preprocessing script
├── unet_model.py             # Model architecture code (from step 1)
├── potential_dataset.py      # Dataset loading code (from step 2)
├── train.py                  # Training script (from step 3)
├── evaluate.py               # Evaluation script (from step 4)
└── demo.py                   # Demo script (from step 5)
└── mapf_simulation_xxx.py                   # MAPF simulation script (from step 5)
