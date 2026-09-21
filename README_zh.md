# AgentJev-0.6B

[English](README.md) | [简体中文](README_zh.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen3--0.6B-green.svg)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![Benchmark](https://img.shields.io/badge/Benchmark-Typed_Decisions_79.25%25-orange.svg)](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
[![Latency](https://img.shields.io/badge/Latency-~50--100ms-purple.svg)](#推理性能与时延)

**专为 AI Agent 打造的 0.6B 并行 System One（快速直觉决策）模型：输入任意状态与问题，直接输出经过概率校准的连续分布，零输出 Token 生成。**

---

## ⚡ 核心定位与背景

在构建 AI Agent（如代码修复 Agent、运维排障机器人、工作流引擎）时，业界通常使用 27B~70B+ 的大型生成式模型自回归生成数百字来做简单的逻辑分支判断（例如：*“测试全通过了吗？”*、*“下一步该调哪个工具？”*、*“这个操作安全吗？”*）。这种方式存在明显痛点：
- **时延高**：每个决策步自回归生成 Token 需要数百毫秒甚至数秒；
- **成本昂贵**：大量 Token 预算白白浪费在单一的布尔值或多选分支上；
- **易出错**：容易出现格式解析失败（JSON 幻觉）和未校准的置信度。

**AgentJev-0.6B** 旨在充当 Agent 的 **“大脑前额叶 / 快速神经反射弧（System One）”**。输入非结构化的业务状态（代码 Diff、错误堆栈、工单多轮对话、数据表）与结构化决策问题，AgentJev 仅需**单次前向传播（~50ms）**即可输出连续、高保真的概率分布。

<p align="center">
  <img src="agentjev_reflex_demo.gif" alt="AgentJev 快速直觉反射 vs 传统大模型自回归对比" width="100%" />
</p>

---

## 🏆 官方 Benchmark 评测成绩

模型在权威公开基准 **[Typed Decisions Benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)** (`LocalLLaMA/typed-decisions`) 独立测试集（**400 个全新业务案例，2,000 道结构化决策题**）上进行了全量同题测试：

| 模型 | 模型定位 | Top-1 准确率 | 软交叉熵 (CE) ↓ | Brier 概率误差 ↓ | ECE 校准误差 ↓ | 等级误差 (MAE) ↓ | 单案例平均时延 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AgentJev-0.6B (本项目)** | **Specialist** | **79.25%** (1585/2000) | **0.8494** | **0.0448** | 0.1687 | **0.2096** | **~100 ms** |
| Laya 官方专用微调版 | Specialist | 77.00% (1540/2000) | 0.8844 | 0.0615 | 0.2170 | 0.2423 | ~120 ms |
| TypeSafe Jev 1.13.0 | Commercial | 72.70% (1454/2000) | — | 0.1480 | 0.1440 | 0.3910 | 710 ms |
| ModernBERT-base (149M) | Specialist | 64.60% | — | 0.1190 | 0.1790 | 0.4440 | 349 ms |
| MiniLM-L6 (22M) | Specialist | 58.70% | — | 0.1430 | 0.1080 | 0.5150 | 22 ms |
| 未微调 Phase 4 基线 | Specialist | 38.70% (774/2000) | 1.2817 | 0.2577 | 0.1050 | 0.7062 | ~100 ms |
| 标签频率盲猜 (Prior) | Reference | 47.00% | — | 0.1890 | 0.0880 | — | 0 ms |
| 均匀随机盲猜 (Uniform) | Reference | 30.80% | — | 0.2380 | 0.1690 | — | 0 ms |

> **统计显著性检验**：经 2,000 次 Case 级别的 Bootstrap 重抽样检验，AgentJev-0.6B 相对 Laya 取得 **+2.25% 的净胜优势**，95% 置信区间为 `[+0.60%, +3.80%]`（区间下界严格大于 0，$p < 0.05$）。

### 细分业务领域准确率
- **发票核销与财务对账** (500 题)：**86.20%** (*Laya: 81.20%*)
- **客户服务工单分派** (500 题)：**82.20%** (*Laya: 76.40%*)
- **安全突发事件调查** (500 题)：**76.80%** (*Laya: 77.60%*)
- **Agent 执行链路观测** (500 题)：**71.80%** (*Laya: 72.80%*)

---

## 🌟 核心架构与技术亮点

1. **零 Token 解码（Zero Output Tokens）**
   - 不进行自回归逐字生成，直接通过打分头与分类头投影输出概率分布。
2. **2048 Tokens 超大上下文容量**
   - 相比传统 1024 限制的决策编码器**容量提升一倍**，完整容纳大型 Git Diff、长报错堆栈与多轮客户对话，无截断损失。
3. **原生共享前缀 KV 复用（Shared Prefix Cache）**
   - 面临 64 个甚至 255 个候选项时，环境状态只需在骨干网络中前向计算 1 次。实测多候选负载下时延从 **610ms 减半至 299ms**。

<p align="center">
  <img src="agentjev_shared_prefix.gif" alt="AgentJev 共享前缀 KV 复用加速对比" width="100%" />
</p>

4. **三种标准决策原语支持**：
   - **`Boolean`**：命题真假判定（支持自定义 True/False 语义准则）。
   - **`Choice`**：支持 2~255 个动态候选项的多项选择。
   - **`Score`**：支持 2~10 级有序等级评分，并返回连续期望得分 $\sum (i \times P_i)$。

---

## 🚀 快速上手

### 1. 环境准备

```bash
git clone https://github.com/your-org/AgentJev.git
cd AgentJev
pip install -r requirements.txt
```

### 2. 启动本地决策推理服务

```bash
python -m jev_service.server --checkpoint checkpoints/agentjev_v1/best.pt --port 8149
```

服务启动后，浏览器访问 `http://127.0.0.1:8149/` 即可直接打开可视化交互工作台。

---

## 💻 Python Client SDK 使用示例

使用官方提供的 Python 客户端 SDK（`agentjev_client.py`），无缝嵌入现有的 Agent 循环中：

```python
from agentjev_client import AgentJev

# 初始化连接
jev = AgentJev("http://127.0.0.1:8149")

# 1. 布尔门控：单测是否全部通过？
state = {
    "task": "修复 UserAuthService 的空指针异常",
    "test_output": "Tests run: 14, Failures: 1, Errors: 0. 失败用例: test_expired_token"
}

is_done = jev.decide_boolean(
    state=state,
    question="当前任务是否已彻底修复并可安全提交 PR？",
    criteria={
        "true": "所有单元测试通过且编译无误。",
        "false": "仍有失败测试或未满足的需求。"
    }
)
print("允许提交 PR 吗?:", is_done["decision"])
# 输出: 允许提交 PR 吗?: False (未完成置信度: 58.04%)

# 2. 动态多选：下一步最佳动作？
next_action = jev.decide_choice(
    state=state,
    question="Coding Agent 下一步最合理的动作是什么？",
    options={
        "read_failed_test": "查看失败测试用例的源码，确认它期望捕获什么异常。",
        "rewrite_entire_file": "要求大模型推倒重写整个服务文件。",
        "force_commit": "忽略失败测试，强行合并代码。",
        "blind_retry": "不改代码直接重跑单测。"
    }
)
print("推荐动作:", next_action["best_action"])
print("置信优势 Margin:", next_action["margin"])
# 输出: 推荐动作: read_failed_test (概率: 54.3%, Margin: +37.1%)

# 3. 有序等级评分：评估代码改动风险
risk = jev.score(
    state=state,
    question="评估该代码变更对生产环境的回归风险等级：",
    levels=[
        "Level 0: 隔离修改，零外部影响。",
        "Level 1: 轻度风险，单测断言小幅变更。",
        "Level 2: 中度风险，涉及 API 签名变更。",
        "Level 3: 高危风险，可能破坏鉴权一致性。"
    ]
)
print(f"风险评级: {risk['level']} (期望分值: {risk['expected_score']:.2f} / 3.0)")
# 输出: 风险评级: 1 (期望分值: 1.54 / 3.0)
```

---

## 📡 HTTP 接口规范

AgentJev 暴露标准 REST 接口：

`POST /api/evaluate`

```json
{
  "state": "当前代码修改完毕，单测运行：23 个通过，1 个失败。",
  "questions": [
    {
      "id": "is_completed",
      "type": "boolean",
      "question": "测试是否已全部通过？"
    },
    {
      "id": "next_step",
      "type": "choice",
      "question": "下一步应该采取什么动作？",
      "options": {
        "debug_failure": "查看失败断言的具体实现。",
        "submit_patch": "直接提交代码。"
      }
    }
  ]
}
```

---

## 🎯 典型实战落地场景

- **Claude Code 与开发环境实时门控**：作为 `PreToolUse` 钩子运行，在终端执行高危命令（如 `rm`、强制推送、越权访问）或保存代码前进行毫秒级拦截与风险审查。

<p align="center">
  <img src="agentjev_gating_hook.gif" alt="AgentJev 实时安全门控" width="100%" />
</p>

- **Coding Agent 动作路由与单测门禁**：瞬时判定测试是否通过并指导下一步查看错误断言或重试，节省 70%+ 的 Agent 循环 Token 成本。
- **客户工单意图识别与退款分派**：快速分类用户诉求，评估流失风险，并分派至对应专员。
- **财务发票与订单核销**：检测明细金额与交货单差异，防范重复开票。

---

## 🛡️ 数据集防泄露与实验卫生

所有训练与验证严格遵守学术防泄露规范：
- **案例级彻底物理隔离**：官方 1,200 个训练案例与 400 个测试案例完全按 Case ID 和状态哈希做排他切分，两集交集严格为 0；
- **测试集严格封存**：测试集在第 600 步选定最优模型之前从未被读取，未参与任何超参调试或早停监控；
- **去标签化**：输入文本严格剥离了任何目标标签、黄金答案与内部任务标记。

---

## 📜 开源协议

本项目基于 [Apache-2.0](LICENSE) 开源协议发布。
