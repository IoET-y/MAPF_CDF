your_project/
├── src/
│   ├── env/
│   │   ├── map_loader.py         # 负责地图/障碍物读写
│   │   ├── multi_agent_env.py    # 多智能体环境主类
│   │   └── ...
│   ├── active_matter/
│   │   ├── forces.py             # 各种势函数定义
│   │   └── active_matter_sim.py  # 动力学仿真
│   ├── diffusion/
│   │   ├── model.py              # Diffusion 模型网络结构
│   │   ├── schedule.py           # 噪声日程相关
│   │   ├── training.py           # 训练循环
│   │   └── sampling.py           # 推断/去噪采样
│   ├── integration/
│   │   ├── offline_planner.py    # 离线调用 diffusion 生成轨迹 + Active Matter 局部执行
│   │   ├── online_planner.py     # 在线耦合 + 动态环境重新规划(可选)
│   │   └── ...
│   └── utils/
│       ├── collision_check.py    # 碰撞检测
│       ├── visualization.py      # 可视化工具
│       └── metrics.py            # 统计指标
├── data/
│   ├── maps/                     # 各种地图
│   ├── expert_trajs/             # 存放 A*, CBS 等专家轨迹
│   ├── active_matter_trajs/      # 存放 Active Matter 生成的轨迹(可选)
│   └── ...
├── experiments/
│   ├── train_diffusion.py        # 训练入口脚本
│   ├── run_simulation.py         # 主测试脚本
│   └── ...
├── docs/
│   └── ...                       # 设计文档、实验报告等
└── README.md
