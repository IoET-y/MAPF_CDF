medium-mazes-seed-2335_agents8.npz

nohup python train.py \
    --dataset-dir ../MAPF-GPT/prepared_agent_centric_dataset_v4 \
    --output-dir training_output \
    --val-split 0.10 \
    --epochs 100 \
    --batch-size 5120 \
    --lr 0.001 \
    --base-channels 64 \
    --obs-radius 5 \
    --device cuda \
    --num-workers 4 \
    --checkpoint-freq 10  > output_train_2.log 2>&1 &
    # Add --resume-checkpoint path/to/checkpoint.pth to resume


python evaluate.py \
    --dataset-dir prepared_agent_centric_dataset_v3/test \
    --checkpoint training_output/best_model.pth \
    --batch-size 128 \
    --obs-radius 5 \
    --base-channels 64 \
    --device cuda


python demo.py \
    --npz-file ../MAPF-GPT/prepared_agent_centric_dataset_v3/medium-mazes-seed-2335_agents8.npz \
    --agent-index 3 \
    --checkpoint training_output/best_model.pth \
    --base-channels 64 \
    --device cpu # Can run demo on CPU even if trained on GPU

python mapf_simulation_lpf.py --map-name medium-mazes-seed-3999 --num-agents 24 --checkpoint  training_output/best_model.pth 
python mapf_simulation_lpf.py --map-name Berlin_1_256_00 --num-agents 192 --checkpoint  training_output/best_model.pth 

medium-mazes-seed-2335


python mapf_simulation_lpf.py \
    --controller lpf \
    --map-name "medium-mazes-seed-0001" \
    --num-agents 8 \
    --obs-radius 5 \
    --checkpoint path/to/your/6_channel_model.pth \
    --input-channels 6 \
    --base-channels 64 \
    --decoder-strategy probabilistic \
    --temperature 0.8 \
    --resolver-strategy random \
    --device cuda

python mapf_simulation_slbfs.py \
    --controller slbfs \
    --map-name "medium-mazes-seed-3999" \
    --num-agents 192 \
    --obs-radius 5 \
    --resolver-strategy random \
    --device cpu # SL-BFS doesn't use GPU model

    
Berlin_1_256_00