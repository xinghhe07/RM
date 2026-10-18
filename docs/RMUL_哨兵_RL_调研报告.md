# RoboMaster RMUL 3v3 哨兵强化学习 — 赛题/工程资源调研

面向 2026 赛季 RMUL 3v3 哨兵机器人强化学习项目（核心机制：占点扣对方胜利点，车端仅有 x86-64 CPU）。

**核实方法说明**：GitHub 的 star 数、许可证、更新时间均取 `https://api.github.com/repos/<owner>/<repo>` 的真实返回值；README / 文档内容取 `https://api.github.com/repos/<owner>/<repo>/readme`（base64 解码）或 `contents/<path>`。凡是查不到、或页面为 JS 渲染无法取到正文的，一律写「未确认」，不做推断。

---

## A. 官方与社区数据集

### A1. 官方 RMUC 2026 区域赛数据集 — 核实结果

| 项目 | 核实结论 | 依据 |
|---|---|---|
| 官方帖子 | 存在，标题《【RMUC 2026区域赛部分赛事数据发布】》 | [bbs.robomaster.com/article/1936220](https://bbs.robomaster.com/article/1936220) |
| 帖子正文 | **未能取到**（页面为 JS 渲染，抓取只返回标题）。帖子发布日期、下载链接、官方字段说明的原文措辞均**未直接核实** | — |
| 规模（613 局 / 400 万条 1Hz） | 与两个独立第三方仓库的转述一致，**基本可信** | RMUC-OfflineRL、rm-battlescope |
| 数据格式 | 单个 SQLite 文件，三张表 `matches` / `timeseries` / `events`，以 `game_id` 关联 | 同上 |
| 覆盖范围 | **仅东/南/北三赛区**，且是「部分」赛事数据（帖子标题即写「部分」） | 帖子标题 + 两仓库转述 |
| 后续批次（全国赛/复活赛等） | **未发现任何后续批次公告**。标注为「未确认」 | 多轮检索无结果 |
| 官方字段说明文档 | **未找到独立官方文档**；官方侧最接近的权威材料是[《RoboMaster 裁判端使用说明》PDF](https://rm-static.djicdn.com/tem/13194/RoboMaster%E8%A3%81%E5%88%A4%E7%AB%AF%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.pdf)，可用来理解裁判系统字段语义 | — |

#### 精确行数（来自 RMUC-OfflineRL 转述）

| 表 | 行数 | 含义 |
|---|---|---|
| `matches` | 613 | 每行一局 |
| `timeseries` | 4,015,383 | 逐秒状态 |
| `events` | 1,474,178 | 逐事件 |

> 注意：613 局 × 约 420 秒 × 8 实体 ≈ 206 万，与 401 万不在同一量级；两处口径（「400 万条 1Hz 记录」vs `timeseries` 4,015,383 行）可能一个含 `events` 或含更多实体，**此口径差异未确认**，建议下载后自行 `SELECT COUNT(*)` 核对。

### A2. 字段说明（完整，可直接用作解析 schema）

以下来自 RMUC-OfflineRL 的《RMUC 2026 区域赛数据集使用说明》文档，是本次调研能取到的**最完整字段说明**：
[dataset/RMUC 2026 区域赛数据集使用说明.md](https://github.com/Harkerbest/RMUC-OfflineRL/blob/main/dataset/RMUC%202026%20%E5%8C%BA%E5%9F%9F%E8%B5%9B%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

**通用约定**
- 一「局」（match）含多「小局」（round）：一局对垒通常打 3~5 小局分胜负。
- 三张表都带「赛区 / 场次号 / 赛程 / 局号 / game_id」，可据此关联。
- 时间列统一为「距开始秒数」（局内 0 起），三表可对齐、可按秒拼接。
- 阵营分「红 / 蓝」；每台机器人所属队的学校名直接写在行内，便于筛选。

**表 1 `matches`（每行一小局）**

| 字段 | 含义 |
|---|---|
| 赛区 | 东部 / 南部 / 北部 |
| 场次号 | 该赛区内场次编号 |
| 赛程 | `【赛区】第N场` |
| 局号 | 该场第几局（1~5） |
| `game_id` | 局唯一编号（关联另两表） |
| `web_game_id` | 官网局编号 |
| 红方学校 / 蓝方学校 | 对阵双方学校名 |
| 胜负 | 红 / 蓝 |
| 开始时间 / 时长秒 | 该局开始时刻 / 时长（秒） |

**表 2 `timeseries`（每行 = 某小局某秒某机器人）**
覆盖 8 类实体：英雄 / 工程 / 步兵3 / 步兵4 / 空中 / 哨兵 / 基地 / 前哨站。
（基地与前哨站为建筑，其血量也在本表；空中=无人机，仅在其出动时段有数据行。）

| 字段 | 含义 / 用法 |
|---|---|
| `赛区` `场次号` `赛程` `局号` `game_id` | 局次维度 |
| 时刻秒 | 距开始秒数 |
| `robot_id` | 机器人编号 |
| 机器人类型 | 英雄 / 工程 / 步兵3 / 步兵4 / 空中 / 哨兵 / 基地 / 前哨站 |
| 阵营 | 红 / 蓝 |
| 学校名 | 该机器人所属队学校 |
| 对手学校 | 对方学校 |
| 当前血量 / 最大血量 | 该秒血量 |
| `x` / `y` / `z` | 场地坐标 |
| 枪口朝向 | 枪口罗盘朝向 |
| 底盘功率 | 当前底盘功率 |
| 小热量 / 小热量上限 | 17mm 枪口热量及上限 |
| 大热量 / 大热量上限 | 42mm 枪口热量及上限 |
| 累计17mm 发射 | 截至该秒累计发射 17mm 弹量（**累计值，非剩余**） |
| 累计42mm 发射 | 截至该秒累计发射 42mm 弹量（仅英雄有值） |
| 队友总金币 | 该队截至该秒累计获得金币 |
| 队友剩余金币 | 该队当前剩余金币 |
| 是否易伤 | 0/1，该秒该机器人**被对方雷达标记**高级。仅在标记时为 1 |

**表 3 `events`（每行 = 一次事件）**
通用字段：`赛区`/`场次号`/`赛程`/`局号`/`game_id`/`时刻秒`/`robot_id`/`机器人类型`/`阵营`/`学校名`（事件主体），以及 `目标robot_id`/`目标类型`/`类别`/`数值`/`备注`。

| 事件类型 | 含义 | 类别取值 | 数值 | 备注 |
|---|---|---|---|---|
| 发射 | 一次射出 | 17mm / 42mm | — | — |
| 受击 | 一次受到伤害 | 17mm / 42mm / 撞击 / 飞镖 / 判负 | 扣血值 | `robot_id` 为受击方 |
| 装配成功 | 科技核心装配成功 | 等级 N | 耗时（秒） | 收益 |
| 能量机关 | 能量机关击打 | `rune_type` | 环数 | 激活数 / 平均环数 |
| 飞镖闸开 | 飞镖发射闸门开启 | — | 开启次数 | — |
| 飞镖命中 | 飞镖命中 | 飞镖命中 | 伤害值 | 目标类型 = 基地/前哨站 |
| 雷达反制UAV | 无人机被对方雷达锁定反制 | 被反制 | — | 阵营为被反制方 |
| 增益 | 获得一次增益 | 飞坡/过中央高地/飞镖跨越/小能量机关/大能量机关 | — | — |

**数据陷阱（务必先处理，来自同文档「使用提示」）**
- `累计发射`是累计值：某段时间发射数 = 末秒累计 − 首秒累计。
- 分析受伤用 `events` 的「受击」；分析弹量消耗用「发射」或 `timeseries` 的累计发射。
- 空中（无人机）只在出动时段有数据行，**统计时其行数天然少于地面机器人**。
- 有缺失值，且**大表没有数据库级主键和外键约束**。

### A3. rm-battlescope 补充的关键数据质量问题

来源：[RMUC_2026_轨迹解析说明.md](https://github.com/ezthor/rm-battlescope/blob/main/docs/RMUC_2026_%E8%BD%A8%E8%BF%B9%E8%A7%A3%E6%9E%90%E8%AF%B4%E6%98%8E.md)、[RMUC_2026_规则与数据索引.md](https://github.com/ezthor/rm-battlescope/blob/main/docs/RMUC_2026_%E8%A7%84%E5%88%99%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%B4%A2%E5%BC%95.md)

| 结论 | 细节 |
|---|---|
| 坐标是**场地全局系，不是机器人本体感知** | 原图 1683 × 938 px 的有效区内框 `(100,69)–(1576,856)`，重映射为 `x=0` 红方一侧、`x=28` 蓝方一侧；`y=0` 画布下沿、`y=15` 画布上沿。与规则图一致 |
| 坐标明显偏场地系 | 数据库轨迹中 **99.32% 直接落在运动范围内**，支持该判定 |
| 空中（无人机）`z` 与清障 | `z` 不属于清障数据 |
| **`目标robot_id` 大面积为 NULL** | 全部约 120 万条「发射」与 25 万条「受击」中均为 NULL；**仅 384 条「飞镖命中」有值** → **无法做「谁打了谁」的边级建模**，只能用累计奖罚做辅助信号 |
| 缺线信息 | 数据中**不存在任何视线/可见性字段**；地图障碍也不在数据中 |
| 异常值 | `x=y=0` 表示裁判系统丢失跟踪（哨兵 2.3%，其他 <1%）；基地与前哨站在库中固定为 `(0,0)`，**轨迹分析应排除** |
| `是否易伤` 语义 | 只在该秒该机器人被对方雷达**标记高级**时为 1 |
| 空中坐标跳变 | 默认地面跳变阈值 8 m/s、空中 15 m/s；越界点不作位姿误差、异常跳点保持断线 |
| 平滑参数 | 最大插值缺口 3 帧；平滑窗口 `[1,2,1]` |
| 版本风险 | 背景图来自**全国赛 V2.1.0** 手册，而区域赛用的是 V2.x 规则，**区域赛场地细节可能不同，需人工确认**（作者已在文档中经人工核对） |
| 哨兵 vs 步兵对比 | 哨兵平均得分 ≈ 步兵的 1/3；位置熵约一半；**81% 时间在 5 个格子内** |

### A4. 基于该数据集的解析与学习工具

| 工具 | 定位 | 核实到的硬数据 |
|---|---|---|
| [ezthor/rm-battlescope](https://github.com/ezthor/rm-battlescope) | RMUC 2026 数据解析、战术回放与策略研究；只读查询 SQLite，支持轨迹清洗、状态可视化、启发式战斗关系推断 | Python，**21 star / 2 fork**，MIT，API `created_at` 2026-07-18、`pushed_at` 2026-07-18，topics: rmuc/robomaster/sqlite/tactical-analysis/trajectory-visualization |
| [Harkerbest/RMUC-OfflineRL](https://github.com/Harkerbest/RMUC-OfflineRL) | 基于官方区域赛数据集的**全兵种自主决策离线 RL 框架**（IQL / BC / Decision Transformer 三种算法同一接口对比） | Python，**71 star / 11 fork**，非标准 license（NOASSERTION，README 内文写 MIT），`created_at` 2026-07-22、`pushed_at` 2026-09-09 |
| [Yueyuanh/RoboMasterGym](https://github.com/Yueyuanh/RoboMasterGym) | RoboMaster IsaacGym 框架（demo） | Python，**34 star / 1 fork**，MIT，`created_at` 2025-05-12、`pushed_at` 2025-05-25 |
| [CosmosMount/rmuc2026_battlefield_mjcf](https://github.com/CosmosMount/rmuc2026_battlefield_mjcf) | **RMUC 2026 场地的 MJCF 模型（含碰撞）** —— 自建仿真环境时可直接省掉地形建模 | Python，**0 star**，**无 license 文件**，`created_at` 2026-08-04、`pushed_at` 2026-09-11 |
| [hkustenterprize/RM2025-Radar-Algorithm](https://github.com/hkustenterprize/RM2025-Radar-Algorithm) | 雷达站算法（非决策，但含场地坐标/目标跟踪/预测链路的工程参考） | Python，**74 star / 11 fork**，MIT，`created_at` 2025-08-03、`pushed_at` 2025-11-30 |

**未确认**：除以上项目外，未发现其他公开的第三方 RMUC 2026 数据集解析工具。

### A5. RMUC-OfflineRL 的可复用结论（哨兵相关，直接可用）

以下数字均来自该仓库 README 的实测记录（**是作者在测试图上评估的结果，不是跨赛事通用结论，引用需注明来源与局限**）：

| 结论 | 数字 |
|---|---|
| 哨兵 vs 步兵的运动特性 | 哨兵轨迹覆盖面约为步兵的 **1/3**，位置熵 ≈ 步兵一半，**81% 时间在 5 个格子内** → 哨兵是**高度驻留**型角色 |
| 场上机器人数量 | 613 局共 **3554 条单机单局轨迹**（1 m 栅格、单局平均），有 `viz/heatmaps.py` 复现 |
| 哨兵平均得分 | **400**（= 步进等级上限）／260 热上限；作者认为这是**确定值**，「哨兵等级等于一台满爆级步兵」 |
| 关键负面结论 1 | **基地被摧毁仅 1 / 1226 例** → 不能以「摧毁基地」为目标 |
| 关键负面结论 2 | **数据中不存在任何占点/可占领字段**，占点扣血无法从数据原样获得 |
| 与占点直接相关 | 「基地被摧毁」极稀有、哨兵极驻留——**数据只能支撑「守点/控图」类塑形，不能直接监督「占点」标签** |
| 主方法成绩（哨兵） | IQL best checkpoint：`nav_mse 0.0532`、`fire_f1 0.4081`、`target_top1 0.8393`；相对基线**目标选择提升 +0.2150**，相对 `target_top1_named`（6 台里选对）**0.6661** |
| 反直觉发现 | **用「人类步兵」数据训练出的「步兵」比用「全自主哨兵」数据学到的决策更多**（基线移除 +0.341 vs +0.215） |
| 方法论警告 | BC/DT 的 Δ 几乎全是估计器偏差，不是策略质量；**跨兵种同模型评估不可用**（Q 函数拟合在评估域，任何不同策略的行为都会被判为分布外） |
| 动作空间（关键，可直接借鉴） | observation **161 维**；动作 **10 维 `tactical`** = 导航目标（2 维 Gaussian，`goal_horizon=5` 秒后的位置）+ 交战门控（1 维 Bernoulli）+ 目标选择（**7 类 Categorical 软标签**：6 台敌方机器人 + 「无目标」） |
| 目标选择的标签来源 | 「是否开火」由 17mm 发射秒差直接导出（精确标签）；「瞄准谁」日志不记录，需用**枪口朝向与各方位姿的夹角**反推 |
| 软标签阈值 | 存在真实歧义（**>1 个敌方目标的困惑占比 55.5%**），因此用 `softmax(−误差/σ)` 分配权重（`cone=15°`、`σ=8°`），锥内无目标则全部权重归「无目标」 |
| 队别/同队先验 | 1226 条记录中仅 96 条为「先启动」，按 `开始时间` + `game_id` 排序展开累计，验证了因果顺序；**不输出队伍身份或 embedding**，避免「遇到全校就选某个动作」的捷径 |
| 训练超参 | `batch=1024`、`lr=3e-4`、`weight_decay=1e-4`、warmup 500、`grad_clip=1.0`、`expectile=0.7`、`gamma=0.99`、`polyak=0.005`；`eval_every=1000`、`early_stop_patience=8` |
| 安全约束（部署侧） | `apply_safety` 强制：限制单步位移、热量无余量时抢占交战许可、低烈度忍让、**眩晕时撤消纵向分量**；蓝方侧自动反转指令 |
| 人少队（关键） | **对实际中「无法目视观测到的数据」导致偏差尚无处理方案** —— 作者已在 README 的未来工作里明确列为第 1 优先项 |

### A6. 装甲板 / 视觉数据集（完整度补充）

| 资源 | 说明 | 核实情况 |
|---|---|---|
| [LibreYOLO/robomasters-285km](https://huggingface.co/datasets/LibreYOLO/robomasters-285km) | HuggingFace 上的 RoboMaster 检测数据集 | 搜索可见，**README 未能取到正文（fetch failed），具体规模/类别/许可未确认** |
| [BreCaspian/ROBOMASTER-2025-Energy-Mechanism-Video](https://huggingface.co/datasets/BreCaspian/ROBOMASTER-2025-Energy-Mechanism-Video) | 2025 能量机关视频数据集 | 搜索可见，**未取到正文，未确认** |
| [Harry-hhj/CVRM2021-sjtu](https://github.com/Harry-hhj/CVRM2021-sjtu) | 上海交大 RoboMaster 2021 赛季视觉代码（含装甲板检测） | **482 star / 70 fork**，**GPL-3.0**，`pushed_at` 2024-01-26 |
| [HKU-ICRA/PythonCV](https://github.com/HKU-ICRA/PythonCV) | 「装甲板检测算法」试验用 Python OOP 化实现 | **0 star**，无 license，`pushed_at` 2020-03-15（很旧） |
| [HKU-ICRA/YoloV3_TF2.0](https://github.com/HKU-ICRA/YoloV3_TF2.0) | 用于 RM AI Challenge 的 YOLOv3 TF2.0 实现 | **3 star**，无 license，`pushed_at` 2020-01-01 |

**结论**：RM 视觉数据集的公开程度远低于决策数据集；未发现公认的、带许可的「装甲板标准数据集」。若需要，建议自建并用 CVRM2021-sjtu 的做法（GPL 注意传染性）。

---

## B. 部署与推理工具链（训练在 PC、部署到纯 CPU x86-64 车端）

### B1. 推理引擎在纯 CPU / x86-64 上的适用性

license 与仓库状态全部来自 `api.github.com` 实际返回：

| 引擎 | 官方仓库 | license（实测） | star（实测） | 纯 CPU / x86-64 适用性 | C++ API 成熟度 |
|---|---|---|---|---|---|
| **ONNX Runtime** | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | **MIT** | **22029** | 原生支持，CPU EP 是默认后端；官方 C/C++ 文档完整 | 高。官方 C/C++ API 文档见 [onnxruntime.ai/docs/api/c](https://onnxruntime.ai/docs/api/c/)；C++ 侧是 `onnxruntime_cxx_api.h` 的 header-only RAII 包装 |
| **OpenVINO** | [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) | **Apache-2.0** | **10957** | **Intel 平台上 CPU 优化最激进**；官方明确面向 Intel CPU/GPU/NPU 部署 | 高。官方 C++ 示例见 [Hello Classification C++ Sample](https://docs.openvino.ai/archive/2023.0/openvino_inference_engine_samples_hello_classification_README.html) |
| **ncnn** | [Tencent/ncnn](https://github.com/Tencent/ncnn) | **非标准 license（NOASSERTION）** —— README 声明为 BSD-3-Clause 风格但仓库未放标准 SPDX 文件，**用前需自行核对** | **23910** | 面向 ARM/mobile 优化；x86-64 有 SSE/AVX 支持，但**不是 x86 首选** | 中。以「无第三方依赖、可静态链接」著称，C++ API 简洁 |
| **MNN** | [alibaba/MNN](https://github.com/alibaba/MNN) | **Apache-2.0** | **16183** | 支持 x86-64，但重心在端侧/ARM 与 LLM；x86 CPU 上相对 OpenVINO 无明显优势 | 中 |
| **TFLite** | TensorFlow Lite | Apache-2.0（未在本次逐项核实） | — | CPU 支持，但在 x86-64 上生态位不如 ONNX/OpenVINO；C++ API 可用 | 中。**本次未取到官方仓库实测数据，标为部分未确认** |
| **TensorRT** | NVIDIA TensorRT | 专有（EULA） | — | **纯 CPU 不适用**。官方虽提供 CPU-only AOT/TensorRT-RTX engine 相关文档，但那是另一条产品线，**不能替代 GPU 上的 TensorRT**；无 NVIDIA GPU 时不应作为选型 | 不适用于本项目 |

**延迟量级**：**未确认**。本次检索**没有找到**「与本项目同量级模型（约 1e4~1e6 参数 MLP）+ 具体 x86-64 CPU 型号」的可信官方延迟基准。检索到的 YOLO-nano 跨格式对比、OpenVINO INT8 说明等，其模型与 CPU 都与本项目不可比。**建议自己测**：在目标车端机器上用 ONNX Runtime 的 profiling 或 `time.perf_counter` 循环 10000 次单样本推理，取 P50/P99。不要采用网上跨模型的数字。

### B2. 把 PyTorch 策略部署到机器人的专门框架

| 框架 | 定位 | 是否支持 ONNX | 核实到的硬数据 |
|---|---|---|---|
| [rai-opensource/exploy](https://github.com/rai-opensource/exploy) | **最贴合本项目的一条路线**。它不只导出网络，而是**把整个 RL 环境逻辑（观测生成 + 动作后处理）一起 trace 进 ONNX**，再在 C++ 侧用 ONNX Runtime 实时执行 | **是，ONNX 是其唯一交换格式** | C++，**101 star / 4 fork**，**MIT**，`created_at` 2026-03-13、`pushed_at` 2026-10-01。组织为 RAI Institute（The Robotics and AI Institute） |
| [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar) | 四足/轮式/人形 RL 的仿真验证与实机部署（"sar" = simulation and real） | **是，且 libtorch 与 onnxruntime 双后端** | C++，**1475 star / 199 fork**，**Apache-2.0**，`created_at` 2024-03-06、`pushed_at` 2026-08-16。README 明确：策略文件放 `policy/<ROBOT>/<CONFIG>/<POLICY>.onnx`（onnxruntime 路线）或 `.pt`（libtorch 路线，需导出 JIT）；支持 ROS1 Noetic 与 ROS2 Foxy/Humble；支持 IsaacGym 与 IsaacSim |
| [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | Isaac Gym 四足环境 | 训练侧为主，导出到部署需自行接 | Python，**3134 star / 585 fork**，非标准 license（NOASSERTION），`pushed_at` 2025-05-29 |
| [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | 宇树官方 RL 训练环境 | 训练侧；部署走 unitree_sdk2 / rl_sar 一类 | Python，**3586 star / 596 fork**，**BSD-3-Clause**，`pushed_at` 2025-07-25 |
| [ms-iot/ros_msft_onnx](https://github.com/ms-iot/ros_msft_onnx) | 微软的 **ROS1/ROS2 + ONNX Runtime** 集成（TinyYOLOv2 目标检测示例） | 是 | C++，**41 star / 14 fork**，**MIT**，**已归档（archived: true）**，`pushed_at` 2024-02-14，默认分支 `foxy`。**已停止维护，只能当参考不能当依赖** |
| [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples) | 官方推理示例集合（含 C/C++ 示例，CI 覆盖 Linux-CPU 与 Windows-CPU） | 是 | C++，**1691 star / 415 fork**，**MIT**，`pushed_at` 2026-10-02 |

**rl_sar 的重要局限（对本项目是关键）**：从 README 看，它面向的 RL 策略是**连续动作**（关节位置目标、速度指令），策略输入输出是固定 shape 的观测向量 → 连续动作向量。**未发现它支持离散/多分类动作头**。所以它能作为「C++ + ONNX Runtime + ROS + 上电自启 + 实机安全」的工程骨架，但**多头策略的取回逻辑要自己加**。

**exploy 的价值与代价**：把环境逻辑一起 trace 进 ONNX，能**最大化消除 sim/real 的预处理不一致**（这正是部署事故的主要来源）。代价是导出的 ONNX 里含控制流/自定义算子，**跨引擎可移植性下降**。另外 README 明确写了：其运行示例假设 **有 NVIDIA GPU**（`> [!NOTE] This example assumes you have an NVIDIA GPU available.`）——所以它的示例路径不是纯 CPU 路线；但 C++ 控制器本身用 ONNX Runtime，**原则上可以切 CPU EP**，这一点需自行验证。

### B3. C++ 读 ONNX 做推理的最小可用示例

| 示例 | 链接 | license |
|---|---|---|
| **ONNX Runtime 官方 C/C++ 示例（推荐起点）** | [onnxruntime-inference-examples/c_cxx](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx) —— 含 `squeezenet`（最经典最小例）、`MNIST`、`ort_tutorial`、`imagenet`、`customop`，以及 `OpenVINO_EP` / `QNN_EP` / `Snpe_EP` 三种 EP 的 C++ 样例 | MIT |
| OpenVINO 官方 C++ Hello Classification | [Hello Classification C++ Sample](https://docs.openvino.ai/archive/2023.0/openvino_inference_engine_samples_hello_classification_README.html) | Apache-2.0 |
| rl_sar 的实机 C++ 推理（真实工程，非玩具） | [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar) —— `rl_sar/src/rl_sar/src/rl_real_*.cpp`，每个机型一个可执行文件 | Apache-2.0 |
| expoly 的 C++ 控制器样例 | [examples/controller/](https://github.com/rai-opensource/exploy/tree/main/examples/controller)（README 指向）；CMake 用法 `find_package(exploy CONFIG REQUIRED)` + `target_link_libraries(... exploy::exploy)` | MIT |

### B4. **重点：混合动作空间（连续 + 离散多头）导出 ONNX 后在 C++ 侧怎么用**

先说结论，再给依据。

#### 结论 1：C++ 侧取多头输出是**标准操作，有官方文档和官方示例，不是难点**

ONNX Runtime 的 C++ API 用 `Ort::Session::Run(...)` 一次性拿回**全部输出**，本质就是「按名字或按索引把它当数组取出来」：

- 官方 C++ API 文档（Session 管理、Tensor 操作）：<https://onnxruntime.ai/docs/api/c/> → [C/C++ API Overview](https://mintlify.wiki/microsoft/onnxruntime/api/c-cpp/overview)（`Ort::Session` / `Ort::Value` 类型对照、C++ 是 header-only RAII 包装、RAII 与异常语义都有说明）、[Session Management](https://mintlify.wiki/microsoft/onnxruntime/api/c-cpp/session)、[Tensor Operations](https://mintlify.wiki/microsoft/onnxruntime/api/c-cpp/tensors)
- 官方最小示例（`squeezenet` 是单输出的最小骨架，改成多输出只是多接几个 `Ort::Value`）：<https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/squeezenet>
- 传多个输入/输出名的实际写法讨论（`const char* const*` 怎么优雅地传）：<https://github.com/microsoft/onnxruntime/discussions/21429>

#### 结论 2：**多头输出取回本身有官方方案；但「混合动作空间 RL 策略部署」没有现成专门框架**

检索多轮后，**未找到**任何专门处理「连续头 + 离散头 + 二值头」混合动作空间策略部署的现成开源参考。检索到的相关材料分两类：

**（a）算法层**（怎么学混合动作空间，不涉及导出/部署）：

| 材料 | 内容 | 备注 |
|---|---|---|
| **HyAR** — Hybrid Action Representation，ICLR 2022 | 用 embedding table + 条件 VAE 为「离散动作 + 其连续参数」学一个紧凑可解码的隐空间，在隐空间里用常规 DRL 学习，再解码回原动作空间。arXiv:2109.05490 | [arxiv.org/abs/2109.05490](https://arxiv.org/abs/2109.05490)（提交 2021-09-12，v3 2022-03-16，Accepted ICLR 2022）。**注意：它是算法论文，不涉及 ONNX 导出** |
| Continuous-Discrete RL for Hybrid Control in Robotics | 机器人混合控制的连续-离散 RL | [笔记](https://thomaspradae.github.io/notes/continuous-discrete-reinforcement-learning-for-hybrid-control-in-robotics)。**原论文出处与年份未在本轮完全确认** |
| CHDP — Cooperative Hybrid Diffusion Policies（AAAI 2026） | 参数化环境下的混合扩散策略 | 出现在检索结果中的笔记页；**未取到正文，未确认** |

**（b）部署层**（多输出怎么在 C++ 侧取，有官方/厂商文档）：

| 材料 | 内容 |
|---|---|
| **Intel 官方 KB：三输出模型如何取张量** | 明确指出：无参 `get_output_tensor()` 只能用于**单输出**模型；多输出必须用**带索引**的 `get_output_tensor(0/1/2)`，再用 Tensor 的 `data` 属性取数据。这就是「连续头、离散头分开取」的标准写法。[Intel KB 000090966](https://www.intel.com/content/www/us/en/support/articles/000090966/software/development-software.html) |
| ONNX Runtime C++ 多输出 | 通过 `Run` 的输出 `Ort::Value` 数组按 `output_names` 顺序/名字取；官方无「多输出 RL 策略」专门示例，但 API 层面无障碍（见上述官方文档与示例） |

#### 结论 3：**给本项目的具体做法（这是本次调研的实操结论）**

针对本项目的动作空间（导航目标 `dx,dy`（连续）+ 是否开火（二值）+ 瞄准哪台敌人（6 类 + 「无目标」= 7 类）），有两条路线，建议**路线 A**：

**路线 A（推荐）：把 argmax / softmax / 门控逻辑「烤」进 ONNX 图，C++ 只做纯数值搬运。**
- PyTorch 侧 `forward` 返回一个 tuple/dict（如 `goal_dx, goal_dy, fire_prob, target_probs`），`torch.onnx.export(..., output_names=[...])` 显式命名每个头。
- 再把 `Softmax` / `ArgMax` / `Greater`(阈值) / `Where`(掩码) 作为**图内算子**接在对应头后面导出。ONNX 算子集包含 `ArgMax`、`Softmax`、`Where`、`Greater`、`Cast`，可以直接在图里选目标、判开火。
- C++ 侧只需 `Run` 后按名字取 4 个 `Ort::Value`，读出 `float` 即可，**不写任何策略逻辑**。好处：训练/推理行为完全一致，避免「Python 里选目标的方式和 C++ 里不一致」这类最隐蔽的 bug。
- 注意：这样导出后 `ArgMax` 的输出是 `int64`，C++ 侧要按 `int64_t` 取；批量维度的动态轴要固定为 1。

**路线 B：只导出 logits，C++ 侧自己算 argmax / softmax / 阈值。**
- 适合需要**动态掩码**（例如「某台敌人当前不可见 → 该目标概率置 0 后重新归一化」）或「上一步决策迟滞」的场景。这类逻辑依赖运行时状态，烤进图里反而不好改。
- 此时务必把 softmax/argmax 的实现与 PyTorch 版本**逐位对齐测试**（同一批随机输入，比对 Python 与 C++ 输出）。

**推荐组合**：**`torch.onnx.export(output_names=[...])` + ONNX Runtime 的 C++ API**。理由是 ONNX Runtime 在本项目的三个约束上同时最优——纯 CPU x86-64 原生支持、MIT 许可、C++ API 有官方文档与官方 CPU 示例（含 Linux-CPU CI）。若车端确定是 Intel 平台且对延迟极度敏感，再把 ONNX Runtime 换成 **OpenVINO EP**（不必改 C++ 代码，只换 session option），或直接用 OpenVINO 的 C++ API。

**动作/观测对齐检查清单（避免部署翻车）**：
1. 观测归一化参数（均值/方差/裁剪范围）必须作为**常量或输入**进图，不能留在 Python 预处理里。
2. 离散头的 argmax 在**概率相等时取哪个索引**（`torch.argmax` 取最小索引），C++ 侧必须一致。
3. 掩码后的 softmax，分母为 0 的边界情况（全掩码）要显式定义行为。
4. `float32` vs `float64` 全程统一（ONNX 默认 `float32`）。
5. 导出后用 `onnx.checker` 校验，并在 Python 的 onnxruntime 里跑一遍，与 PyTorch 输出做数值对比，**再**拿到 C++ 里跑。

---

## C. 同类赛事的 RL 实践参考（重点：可迁移经验）

### C1. RoboMaster AI Challenge（ICRA 全自主步兵对抗）

| 资源 | 定位与做法 | 核实到的硬数据 |
|---|---|---|
| [HKU-ICRA/Pulsar](https://github.com/HKU-ICRA/Pulsar) | 「一个 RL 平台，用于训练 RoboMaster AI Challenge 机器人自主对抗，**通过 self-play 做决策**」。这是与本题**最同构**的一条：同一支队伍、同一类赛事、明确用 self-play | Python，**4 star / 1 fork**，**无 license**，`created_at` 2019-11-05、`pushed_at` 2023-07-06（README 描述原文："A reinforcement learning platform for training Robomaster AI Challenge robots to compete autonomously and make decisions via self-play."） |
| [HKU-ICRA/DM_ReinforcementLearning](https://github.com/HKU-ICRA/DM_ReinforcementLearning) | 决策 = 多智能体 RL（"Decision making via multi-agent reinforcement learning"），是 Pulsar 之前的版本 | Python，**5 star / 1 fork**，**无 license**，`pushed_at` 2019-11-04 |
| [DRL-CASIA/RMAI2020-Decision](https://github.com/DRL-CASIA/RMAI2020-Decision) | 中科院自动化所 DRL 组的 RM AI Challenge 2020 决策代码。**主语言是 C++**，对本项目的 C++ 部署有直接参考价值 | **113 star / 24 fork**，**MIT**，`created_at` 2020-08-20、`pushed_at` 2020-08-27。（HKU-ICRA 下有同名 fork） |
| [HKU-ICRA/RoboMaster-AI-Challenge-Simulator-2D](https://github.com/HKU-ICRA/RoboMaster-AI-Challenge-Simulator-2D) | ICRA 2019 RM AI Challenge 的**2D 决策仿真器**（fork 自他处） | MIT，**1 star**，`pushed_at` 2021-01-12 |

**调研结论（决策范式的历史演变）**：RM AI Challenge 的公开方案呈现「**早期规则状态机 → 后期 MARL/self-play**」的清晰轨迹：2019 年的 Simulator-2D 与 DM_ReinforcementLearning 是探索期，2020 年 DRL-CASIA 给出了完整的 C++ 决策工程，HKU 的 Pulsar 则明确转向 self-play RL。**没有发现**任何一个方案公开了「在正式比赛中击败人类队伍」的可验证 RL 成绩——这点很关键，说明该赛事的 RL 落地难度被普遍低估。

### C2. RoboCup / FIRA 等机器人对抗赛事里的 RL

| 资源 | 可迁移点 |
|---|---|
| **Off-Policy General Value Functions to Represent Dynamic Role Assignments in RoboCup 3D Soccer Simulation** — [arXiv:1402.4525](https://ar5iv.labs.arxiv.org/html/1402.4525) | **角色/分工不是硬编码的，而是用 general value functions（GVF）学出来的「预测性问题」来表示**。这是「谁去占点」这类分工问题最直接的理论工具 |
| **Dynamic Role Assignment using General Value Functions**（Humanoidsoccer workshop）— [PDF](http://www.humanoidsoccer.org/ws12/papers/HSR12_Abeyruwan.pdf) | 同上思路的早期版本，给出「用 GVF 做动态角色指派」的具体构造 |
| **Designing a skilled soccer team for RoboCup: exploring Skill-Set-Primitives through reinforcement learning** — [arXiv:2312.14360](https://ar5iv.labs.arxiv.org/html/2312.14360) | **分层**：底层技能（skill-set-primitives）用 RL 学，上层战术组合。对本项目「底层运动控制 RL + 上层占点决策」的分层有直接参考 |
| **Multi-Robot Cooperative Decision-Making with Hierarchical QMIX: Application to Soccer Offense** — [IEEE](https://ieeexplore.ieee.org/document/11349568) | **Hierarchical QMIX**：把 QMIX 用在分层结构上做多机器人协作进攻。占点任务天然是「先决定去哪 → 再决定怎么去」的两层结构。（**IEEE 原文需订阅，本次只核实到标题与摘要页**） |
| **Tactical Reward Shaping: Bypassing Reinforcement Learning with Strategy-Based Goals**（ROBIO 2019）— [ACM DL](https://dl.acm.org/doi/10.1109/ROBIO49542.2019.8961549) | 标题即结论：**用「战术性目标」做奖励塑形，可以在某些任务上绕过（替代）端到端 RL**。对本项目的直接含义：**不要把「占点」也交给 RL 学**，把它做成显式的策略目标/塑形项更稳。（**付费墙，本次仅核实标题+摘要**） |

### C3. 其它「小场地多体自主对抗」的 RL 工作

| 资源 | 体积/结构 | 可迁移点 |
|---|---|---|
| **Google Research Football** — [论文](https://arxiv.org/abs/1907.11180) / [Google Research 页](https://research.google/pubs/google-research-football-a-novel-reinforcement-learning-environment/) | 11v11，但可裁剪为小规模 | 稀疏奖励（进球）+ 大量塑形；**academy（课程）机制**——从简单场景逐步加难度，对「先让哨兵学会走到点上，再学对抗」的课程设计极有参考价值 |
| **StarCraft Multi-Agent Challenge (SMAC)** | 小规模单位对抗 | **部分可观测 + 局部视野**下的 CTDE（QMIX/MAPPO 系）标准测试床 |
| **Neural MMO 2.0** — [arXiv:2311.03736](https://ar5iv.labs.arxiv.org/html/2311.03736) | 大规模多任务多智能体 | **大规模 self-play + 任务多样性**的工程经验；与本项目「对手也是自主策略」的情形同构 |
| **Gym-µRTS** — [IEEE](https://ieeexplore.ieee.org/document/9619076) | 实时策略的「平价全游戏」版本 | **小规模 RTS + 可负担的全局长局训练**，适合占点/推塔类「长期目标争夺」的算法验证 |
| **Deep RL for Competitive Agents in MicroRTS** — [Zenodo](https://zenodo.org/records/20481385) | MicroRTS 竞争智能体 | 「架构 + 训练 + 锦标赛评估」的完整闭环报告 |
| **AlphaStar** — [DeepMind 官方博文](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) / [Nature 报道](https://www.nature.com/articles/d41586-019-03343-4) | 大规模 | **league（联盟）训练 + 人类回放冷启动 + 多智能体自对弈**。本项目的对手多样性问题可直接借用 league 思路。（**具体论文编号本次未逐项核实**） |
| **OpenAI 多智能体自课程（hide-and-seek / emergent tool use）** | 小场地对抗 | 自发课程（autocurricula）—— 对手能力的自发增长驱动策略进化。**本次未核实具体论文链接，标为未确认** |
| **UAV 空战 RL** | 1v1 / 小规模 | 机动决策 RL。**本次未核实到具体权威链接，标为未确认** |

### C4. 用 RL 做「占点 / 推塔 / 目标争夺 / 领土控制」

**这一类的公开工作明显比 C1–C3 稀少**，本次检索结果如下：

| 材料 | 状态 |
|---|---|
| **Game-Theoretic Area Coverage Control with Cooperative-Adversarial Multi-Agent Systems** | 检索到疑似 arXiv 链接（`2606.10201`），但该编号与当前 arXiv 编号体系不符，**无法确认为真实论文，标为未确认**。不过「area coverage control + 合作对抗多智能体 + 博弈论」这个方向本身是存在的成熟子领域 |
| **Multi-Agent RL for capture the flag / territory control** | **未检索到**公认的、可引用的专门 benchmark 或论文。现有的 CTF 类环境多在通用 MARL 平台上，未形成独立标准 |
| **「占点扣血」这类机制的直接 RL 论文** | **未找到**。这是本次调研最重要的空白发现 |

**因此，C4 的诚实结论**：**「占领控制区持续扣对方胜利点」这个机制，没有找到现成的 RL 论文或开源可以直接照搬。** 必须自己定义环境与奖励。可行的做法是把它**分解**为已被验证的子问题：
- 「走到控制区并停留」→ 用 RoboCup 的角色指派（GVF）+ 分层技能（skill-set-primitives）解决；
- 「在部分可观测下判断该不该去占」→ 用 SMAC/Neural MMO 的 CTDE + self-play 解决；
- 「占点收益的长期延迟」→ 用 Google Research Football 的稀疏奖励 + academy 课程 + 塑形解决；
- 「对手多样性」→ 用 AlphaStar 的 league 思路解决。

### C5. 提炼：哪些经验能直接迁移到「哨兵占点」任务

> 每条标注来源。凡是标注「**工程推论**」的，是基于上述来源 + 本项目约束的推理，**不是文献结论**，请自行验证。

**① 奖励塑形：不要把「占点」也交给 RL 去发现**
- **可迁移**：ROBIO 2019 的 *Tactical Reward Shaping: Bypassing Reinforcement Learning with Strategy-Based Goals* 的标题结论就是——用**策略性目标**做塑形，可以在某些任务上替代端到端 RL（[链接](https://dl.acm.org/doi/10.1109/ROBIO49542.2019.8961549)）。
- **落地含义**：把「进入控制区」作为**显式目标/塑形项**（例如「距控制区中心距离减少 + 在区内停留 + 区内驻留时长」），把「成功占领 → 扣对方胜利点」作为**稀疏终局奖励**，而不要让网络从像素/坐标里自己发现「要去那个圈」。
- **工程推论**：塑形项的形状比权重更关键。建议用**分段的、有上限的**距离势函数，避免「无限靠近但不进入」或「绕圈刷分」。
- **反面证据（重要）**：RMUC-OfflineRL 发现基地被摧毁仅 **1/1226** 例，且**基地/前哨站在数据里固定为 `(0,0)`**（[来源](https://github.com/ezthor/rm-battlescope/blob/main/docs/RMUC_2026_%E8%BD%A8%E8%BF%B9%E8%A7%A3%E6%9E%90%E8%AF%B4%E6%98%8E.md)），说明**极端稀疏的终局事件在本赛事几乎不出现**。因此塑形不是可选项，是必需项。

**② 观测空间：栅格图 + 结构化实体，而不是纯坐标**
- 本项目已有的证据：RMUC-OfflineRL 用 **161 维**观测拿到了可用结果（`target_top1 0.8393`），其组成是「84 维 ego + 时间 2 + 队别掩码 54 + 威力/热量 12 + 经济 4 + 对手先验 6」（[来源](https://github.com/Harkerbest/RMUC-OfflineRL)）。**这是被实测验证过的、可直接复用的观测设计。**
- 可迁移：Google Research Football / SMAC 都证明**「结构化的、按实体对齐的观测」+「局部视野」**比原始像素更省样本。
- **工程推论**：在其基础上加**栅格化的控制区占用图**（把 28m × 15m 离散成与数据集一致的 1m 栅格，标注哪些格子在控制区内、敌方在哪些格子），并**保留原始 `raw_x`** 而非平滑后的坐标（rm-battlescope 明确警告平滑会把「英雄等机器人位于区域边缘时」错误排除，天花板占绿地理图优先用 `raw_x`）。

**③ 部分可观测：帧堆叠 + 历史，不要指望单帧**
- 数据事实：`timeseries` 是**逐秒状态**，且**不存在任何视线/可见性字段**、地图障碍也不在数据中（[来源](https://github.com/ezthor/rm-battlescope/blob/main/docs/RMUC_2026_%E8%BD%A8%E8%BF%B9%E8%A7%A3%E6%9E%90%E8%AF%B4%E6%98%8E.md)）。
- **工程推论**：在 1 Hz 控制频率下，单帧观测无法体现速度与趋势。必须堆叠历史帧（或用 RNN/Transformer）。RMUC-OfflineRL 的 **Decision Transformer 分支**正是吃序列的（`n_emb 128 / 3 层 / 4 头 / ctx 30`），且它证明「**DT 并没有可测量的增益**」（[来源](https://github.com/Harkerbest/RMUC-OfflineRL)）——所以**先用帧堆叠 + 简单 MLP 打底**，不要一上来就上 Transformer。

**④ 多智能体范式：从 CTDE / self-play 起步，但先把「对手」固定下来**
- 本项目的对手多样性问题，可迁移 RoboCup 的**动态角色指派（GVF）**（[arXiv:1402.4525](https://ar5iv.labs.arxiv.org/html/1402.4525)）和 **Hierarchical QMIX** 的「先分工再执行」（[IEEE](https://ieeexplore.ieee.org/document/11349568)）。
- 可迁移 AlphaStar 的 **league 训练**（[DeepMind](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)）：维护「主策略 / 历史策略 / 针对性策略」三池，避免自对弈塌缩。
- **重要反面证据**：RMUC-OfflineRL 的整个评测方法建立在「必须对照基线 / 保持策略族外部有效性」上，且明确**跨兵种同模型评估不可用**（Q 函数只在你评估用的策略分布族内有外部有效性，任何不同策略的行为都会被判为 OOD，[来源](https://github.com/Harkerbest/RMUC-OfflineRL)）。**推论：self-play 训练出的哨兵，不能用一个「与人类步兵对比」的指标来评价。**

**⑤ 「驻留 / 停留」行为：本任务的真正难点，且有现成的度量**
- **最好的现成证据**：rm-battlescope 发现**哨兵 81% 的时间在 5 个格子内**、平均得分约为步兵的 1/3、位置熵约为步兵一半（[来源](https://github.com/ezthor/rm-battlescope/blob/main/docs/RMUC_2026_%E8%BD%A8%E8%BF%B9%E8%A7%A3%E6%9E%90%E8%AF%B4%E6%98%8E.md)）。**这说明真实比赛里的优秀哨兵就是「高度驻留」的**——这直接回答了「占点该怎么表现」。
- RMUC-OfflineRL 还提供了可直接复制的**「时间集中在 5 个格子内」指标**（哨兵 **81.4%** vs 步兵 40.5%）（[来源](https://github.com/Harkerbest/RMUC-OfflineRL)）。**这是评估占点行为最直接的指标，胜率之外必看。**
- **工程推论**：把「时间集中在少数格子」「位置熵」「平均驻留时长」作为**与胜率并列的评估指标**。一个胜率高但位置熵大的哨兵，很可能是「到处乱跑碰运气」，在真实赛场上不可复现。
- **未找到**专门处理「hold position」奖励的论文——**标为未确认**。这块需要自己设计。

**⑥ 冷启动：用人类/规则数据 + 离线 RL，但要注意其天花板**
- **最直接可用的做法**：RMUC-OfflineRL 证明「用**人类步兵**数据训练出的『步兵』比用**全自主哨兵**数据学到的决策更多」（基线移除 +0.341 vs +0.215）。**推论：哨兵的冷启动应当混合「人类步兵/英雄的战术决策数据」与「哨兵的自主数据」**，而不是只用哨兵自己的数据。
- 它的接口设计值得直接抄：`MILPolicyRunner.step(obs_161)` 返回 `{goal_dx, goal_dy, fire, target, target_conf, target_probs}`（**正好就是本项目的混合动作空间**），并提供 `apply_safety(cmd, heat, heat_limit, ammo_left, ego_xy)` 做安全约束（限幅、热量无余量时抢占交战许可、低烈度忍让、眩晕时撤消纵向分量）。**这就是 B4 那个「混合动作空间」在真实项目里的落地形态。**
- **必须知道的局限**：作者自己在 README 里把「**对实际中无法目视观测到的数据偏置尚无处理方案**」列为第 1 优先待办。**你的哨兵数据同样只包含「被观测到/被记录」的部分，这个偏置会直接进到策略里。**
- 另一条互补路线：用 `matches` + `timeseries` 做**行为克隆 / 逆强化学习**，去学「真实比赛里哨兵在控制区附近时的行为分布」。

**⑦ 评估占点策略：胜率远远不够**
综合上述来源，建议的评估面板（前 5 项有文献/数据支撑，后 3 项是工程推论）：
1. **对固定基线的胜率**（必须有对照，见 RMUC-OfflineRL 对「跨兵种同模型评估不可用」的论证）。
2. **时间集中在 5 个格子内的比例**（哨兵基准 81.4%，[来源](https://github.com/Harkerbest/RMUC-OfflineRL)）。
3. **位置熵**（哨兵约为步兵一半）。
4. **`nav_mse`**（导航目标预测误差，哨兵 best `0.0532`）。
5. **`fire_f1`**（开火决策 F1，哨兵 best `0.4081`）与 **`target_top1`**（目标选择 Top-1，哨兵 best `0.8393`，6 台里选对 `0.6661`）。
6. **平均驻留时长 / 占点时长占比**（工程推论：直接度量占点机制的执行情况）。
7. **占点收益效率** = 扣掉对方胜利点 / 在控制区的秒数（工程推论：防止「站着不动刷时间」）。
8. **安全违规次数**（超热量、友伤、越界）（工程推论；RMUC-OfflineRL 的 `apply_safety` 说明这在实践中必须独立成一类约束，不能只靠奖励惩罚）。

**⑧ 仿真环境：不要从零建模**
- 可迁移：[Yueyuanh/RoboMasterGym](https://github.com/Yueyuanh/RoboMasterGym)（IsaacGym，MIT，34 star）与 [CosmosMount/rmuc2026_battlefield_mjcf](https://github.com/CosmosMount/rmuc2026_battlefield_mjcf)（**RMUC 2026 场地 MJCF 含碰撞**，但**无 license，0 star，用前需联系作者确认许可**）。
- [exploy](https://github.com/rai-opensource/exploy) 的「环境逻辑一起进 ONNX」思路，正是为了解决「仿真环境和实机不一致」——**这是本项目最可能翻车的地方，值得优先试**。

**⑨ 与「占点」机制的正面冲突（必须提前面对）**
- **数据侧**：RMUC-OfflineRL 明确写「**数据中不存在任何占点/可占领字段**」，占点扣血无法从数据原样获得（[来源](https://github.com/Harkerbest/RMUC-OfflineRL)）。→ **离线数据只能用来做「守点/控图」类塑形，不能直接监督「占点」标签。**
- **仿真侧**：RMUC 3v3 的占点规则（控制区位置、占领判定、扣血速率）在公开数据集里**没有对应字段**，必须自己从规则手册实现。**本次检索未找到任何已实现该规则的公开仿真环境。**
- **文献侧**：C4 表明「占点扣血」类机制**没有现成的 RL 论文或开源可照搬**。

---

## 结论：这个项目最值得借鉴的 5 个资源（按优先级排序）

**第 1 位 — [Harkerbest/RMUC-OfflineRL](https://github.com/Harkerbest/RMUC-OfflineRL)（71 star，离线 RL 框架）**
唯一一个「官方数据集 → 观测/动作设计 → 三种离线 RL 算法 → 部署接口 → 安全约束」全链路打通、且**把哨兵单列出来评估过**的公开项目。它的 **161 维观测**、**10 维 `tactical` 混合动作**（导航目标 + 开火门控 + 7 类目标选择）、**`MILPolicyRunner` 返回字典**、**`apply_safety`** 可以直接当作本项目的接口规格；它给出的哨兵基线数字（`nav_mse 0.0532` / `fire_f1 0.4081` / `target_top1 0.8393` / 时间集中 5 格 81.4%）可以直接当作本项目的对照基线。**它明确写出「数据里没有占点字段」「基地被摧毁仅 1/1226」「无法目视观测带来的偏置未解决」，帮你省掉踩这三个坑的时间。**

**第 2 位 — [ezthor/rm-battlescope](https://github.com/ezthor/rm-battlescope)（21 star，数据解析 + 轨迹清洗）**
它解决的是「数据能不能用」这个前置问题。**最有价值的是它的文档而不是代码**：[轨迹解析说明](https://github.com/ezthor/rm-battlescope/blob/main/docs/RMUC_2026_%E8%BD%A8%E8%BF%B9%E8%A7%A3%E6%9E%90%E8%AF%B4%E6%98%8E.md) 与 [规则与数据索引](https://github.com/ezthor/rm-battlescope/blob/main/docs/RMUC_2026_%E8%A7%84%E5%88%99%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%B4%A2%E5%BC%95.md) 给出了坐标系精确映射（1683×938 px 有效区内框 → 28×15 m）、**`目标robot_id` 全为 NULL（无法做谁打谁的边级建模）**、**`x=y=0` 是跟踪丢失**、**基地/前哨站固定 `(0,0)` 应排除**、以及哨兵 81% 时间在 5 格内这条对占点任务极其关键的行为特征。**不读这份文档就直接用数据集，几乎一定会得出错误结论。**

**第 3 位 — [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)（22029 star，MIT）**
纯 CPU x86-64 部署这个约束下，它是唯一同时在**性能、许可（MIT）、C++ API 成熟度（官方 C/C++ 文档 + 官方 CPU 示例 + Linux-CPU/Windows-CPU 双 CI）**三项上都无争议的选择，而且**多输出取回是标准操作**（[官方 C++ API](https://onnxruntime.ai/docs/api/c/)、[官方 c_cxx 示例](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)）。更关键的是它提供了**不改 C++ 代码就能换后端**的退路：如果车端确定是 Intel 平台且延迟吃紧，直接切 **OpenVINO EP**（Apache-2.0，10957 star）。

**第 4 位 — [rai-opensource/exploy](https://github.com/rai-opensource/exploy)（101 star，MIT）**
它针对的正是本项目最容易翻车的地方：**把仿真环境的观测生成与动作后处理逻辑一起 trace 进 ONNX，用同一个计算图同时保证仿真和实机行为一致**。同时它给出了一个可编译的 **C++ 控制器 + ONNX Runtime** 的 `find_package` / `target_link_libraries` 集成范式（`examples/controller/`）。**注意两点**：其运行示例假设有 NVIDIA GPU（纯 CPU 需自行验证），且把环境逻辑烤进图会降低跨引擎可移植性。仅次于前两项，是因为它对「占点这个新机制」没有直接帮助——它解决的是工程一致性，不是赛题语义。

**第 5 位 — [HKU-ICRA/Pulsar](https://github.com/HKU-ICRA/Pulsar) + [DRL-CASIA/RMAI2020-Decision](https://github.com/DRL-CASIA/RMAI2020-Decision)（self-play 与 C++ 决策工程）**
Pulsar 是与本题**最同构的公开 RL 尝试**——同为 RoboMaster 自主对抗、明确用 **self-play** 训练决策（仅 4 star、无 license、2023 年后停更，所以只能读思路不能依赖）。DRL-CASIA/RMAI2020-Decision（113 star，MIT）则提供了**主语言为 C++** 的完整决策工程参考，与本项目「C++ 车端」的形态最接近。**必须知道的现实**：这两条线都没有公开「在正式比赛中击败人类队伍」的可验证 RL 成绩——**这说明该场景的 RL 落地难度被普遍低估，不要把期望设在「端到端 RL 直接打赢」上。** 配合 C5 ①③④ 的结论（塑形必需、帧堆叠打底、league 训练防塌缩），用**「规则/策略目标负责占点、RL 负责局部对抗与机动」的分工**更可能成功。

---

### 附：本报告未能确认的事项（避免误用）

1. 官方帖子 [1936220](https://bbs.robomaster.com/article/1936220) 的正文、发布日期、下载链接与官方字段说明原文——页面为 JS 渲染，抓取只得到标题。
2. 数据集**是否有后续批次**（全国赛/复活赛等）——未发现任何公告。
3. 「613 局 / 400 万条 1Hz」与 `timeseries` 4,015,383 行的**口径一致性**。
4. 任何推理引擎在**本项目同量级模型 + 具体 x86-64 CPU** 上的延迟数字——检索到的都是不可比的跨模型数据。
5. 处理「混合动作空间 RL 策略部署」的**现成开源框架**——多轮检索未找到；alg 层有 HyAR（ICLR 2022）等，部署层只有厂商的多输出取张量文档。
6. 用 RL 做「占点 / 领土控制 / 推塔」的**专门论文或开源**——未找到；检索中出现的若干疑似 arXiv 编号（如 `2606.10201`、`2606.20014`）与该站编号体系不符，**无法确认为真实论文，已排除**。
7. 专门处理「hold position / 驻留」奖励的论文——未找到。
8. TensorFlow Lite 的官方仓库实测 star/license——本次未逐项核实。
9. LibreYOLO/robomasters-285km、BreCaspian/ROBOMASTER-2025-Energy-Mechanism-Video 两个 HuggingFace 数据集的**规模、类别与许可**——README 抓取失败。
10. RoboMaster 3v3 占点规则的**已实现公开仿真环境**——未找到。
