** **Beyond Artificial Potential Field**, Better Planning with dynamic BFS under partial observation
**
**当前方案核心：基于学习的、去中心化的局部势场生成**

方法的核心在于：

1. **局部感知：** 每个 agent 仅依赖其有限范围内的观测信息。
2. **势场作为导航信号：** 不直接规划路径或输出动作，而是学习预测一个“势场”，这个场的梯度（或其他解码方式）能引导 agent 移动。
3. **学习驱动：** 使用神经网络（当前是 U-Net）通过数据（目前是模仿 BFS 产生的目标势场）学习如何根据局部观测生成有效的导航势场。
4. **去中心化执行：** 一旦模型训练好，每个 agent 可以独立地根据自己的观测进行势场预测和动作解码，冲突解决则作为协调机制。

### **1. 核心思想**

- **基于学习的局部势场生成**：
    
    利用神经网络（当前采用 U-Net）从每个 agent 的局部感知数据中预测生成导航“势场”。该势场不仅提供了空间分布信息，其梯度还可用来指导 agent 的具体移动。
    
- **去中心化执行**：
    
    每个 agent 仅依赖自身局部的观测数据进行势场预测和动作解码，减少了全局规划和复杂协调的需求，使得系统具有更高的扩展性。
    

---

### **2. novelty**

- **学习复杂势场函数**：
    
    相较于传统的人工势场法（APF）采用简单函数（如线性吸引、反比排斥），你的方法利用神经网络数据驱动地学习，可应对复杂的障碍布局与狭窄通道，同时在一定程度上学习避免 agent 冲突的策略。
    
- **结合深度学习与经典导航思想**：
    
    将经典势场思想与现代深度学习（图像到图像生成技术）的优势相结合，在部分可观测条件下实现更智能的导航。
    
- **中间表示优势**：
    
    通过局部势场作为决策中间变量，比直接输出离散动作更具空间语义，可为后续动作规划提供更丰富的信息。
    

---

### **3. 科研意义**

- **探索局部-全局问题**：
    
    验证仅依靠局部信息生成的势场能否有效支持全局任务的完成（如长距离导航、全局死锁避免），为分布式系统中局部信息与全局目标的关系提供研究实例。
    
- **神经网络空间推理能力**：
    
    想要展示卷积网络（如 U-Net）在环境拓扑推理、导航策略学习中的潜力，为理解神经网络如何“看懂”空间结构提供了实验依据。
    
- **新范式的探索**：
    
    想为多智能体路径规划（MAPF）领域提出了一种新颖的、基于学习的去中心化方法，可与传统搜索、规则或强化学习方法进行对比，拓宽科研视角。
    

---

### **4. 工程及应用优势**

- **高可扩展性与实时响应**：
    
    由于每个 agent 的决策主要依赖自身局部观测，理论上可以并行推理，适用于大规模系统（例如仓库自动化、无人机蜂群）。
    
- **适应部分可观测环境（当前部分观测设定比较理想，后期肯定要贴近真实观测情况）**：
    
    模型直接以局部观测为输入，无需全局地图重建，天然适合动态变化或信息不全的实际场景。
    
- **实现相对简单**：
    
    对比复杂的全局搜索算法（如 CBS），基于标准 U-Net 和简单的解码机制实现起来更直接，有助于工程实践中的快速部署。
    
- **适用范围广泛**：
    
    除了仓库、无人机群体调度，方法还能应用于交通疏导、大规模人群模拟、游戏 AI 等场景，对需要高反应性但“**容忍路径次优**”的任务具有实际吸引力。
    

---

### **5. 初期实验对比直接在观察窗口使用BFS**
死锁情况还没有能很好解决。当窗口区域障碍物有误导性的时候会失败。
但是总体成功率比直接在观察窗口使用BFS会高。
但是推理时间可能会是BFS的2x


### **6. 总结与展望**

这种基于学习和去中心化思路的 MAPF 方法，主要优势在于：

- 利用深度网络学习复杂、适应性更强的势场函数，
- 在局部感知条件下实现智能且扩展性良好的导航，
- 为动态、部分可观测环境提供了一种更实时、反应迅速的解决方案。

下一步工作改进方向可以包括提升势场预测的准确度、优化动作解码与冲突解决策略，以及考虑更多长期规划要素，进一步提高系统整体性能和鲁棒性。

输入可视化

![image](https://github.com/user-attachments/assets/0945bc89-7747-4cf6-ae8f-43e02f33a12d)



```text
your_project/
├── dataset_configs/          #  map config YAMLs，same as https://github.com/CognitiveAISystems/MAPF-GPT/tree/main/dataset_configs
│   └── 10-medium-mazes/
│       └── maps.yaml....
├── eval_configs/  #same as https://github.com/CognitiveAISystems/MAPF-GPT/tree/main/eval_configs
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
└── mapf_simulation_xxx.py                   # MAPF simulation script (from step 6)
└── slbfs_controller.py        # baseline1 directly use BFS for each step
