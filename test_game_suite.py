"""
AgentJev Reflex & Environment Decision Benchmark
================================================
本脚本复现并运行针对 System One 架构的核心环境决策测试样例：

1. 官方题型契约测试样例 (Customer Service / Refund & Triage)
   - Boolean: "Does the customer request a refund?"
   - Choice: "Which team should handle the request?" (returns vs shipping)
   - Score: "How urgent is the request?" (0=No deadline, 1=Mentioned, 2=Immediate)

2. 迷宫探索场景 (50x50 Maze Exploration)
   - 动作选择: 4 向移动决策
   - 安全感知: 死胡同与碰撞检测
   - 距离评估: 出口距离分级

3. 贪吃蛇存活决策 (Snake Survival)
   - 局部安全与食物追踪决策

4. ViZDoom 射击场景 (Basic & Predict Position 预判射击)
   - 目标对齐开火决策
   - 移动目标火箭提前量与发射时机
"""

import json
import time
from agentjev_client import AgentJev

def run_game_replay():
    jev = AgentJev("http://127.0.0.1:8149")
    print("=" * 75)
    print("  🎮 AgentJev 0.6B 游戏与具身动作环境决策全套复现")
    print("=" * 75)

    # -------------------------------------------------------------
    # 测试集 1：契约样例 (Customer Service / Refund)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【测试集 1：System One 契约样例 - 客户退款与工单流转】")
    print("-" * 75)
    state_refund = "Please return my money today. The shoes arrived with torn fabric and do not fit at all."
    print(f"▶ 输入状态 (State): \"{state_refund}\"")

    # 1.1 Boolean
    res_b = jev.decide_boolean(
        state=state_refund,
        question="Does the customer request a refund?",
        criteria={
            "true": "The customer explicitly asks for money back.",
            "false": "The customer makes no request for money back."
        }
    )
    print(f"\n1. [Boolean] 客户是否明确要求退款?")
    print(f"   ✓ 判定: {'【是 (True)】' if res_b['decision'] else '【否 (False)】'}")
    print(f"   ✓ 退款概率: {res_b['prob_true']*100:.2f}% | 非退款概率: {res_b['prob_false']*100:.2f}% (时延: {res_b['wall_ms']}ms)")

    # 1.2 Choice
    res_c = jev.decide_choice(
        state=state_refund,
        question="Which team should handle the request?",
        options={
            "returns": "Handles requests for money back, product return and customer refunds.",
            "shipping": "Handles delivery tracking, courier delays and warehouse logistics.",
            "billing": "Handles invoice generation and corporate tax credit.",
            "general_inquiry": "General FAQ without concrete action."
        }
    )
    print(f"\n2. [Choice] 该工单分派给哪个部门处理?")
    print(f"   ✓ 最佳分派: 【{res_c['best_action']}】 ({res_c['description']})")
    print(f"   ✓ 概率: {res_c['probability']*100:.2f}% (领先次选 Margin: {res_c['margin']*100:.2f}%, 时延: {res_c['wall_ms']}ms)")
    print(f"   ✓ 分布: {json.dumps({k: round(v*100, 2) for k, v in res_c['distribution'].items()})}")

    # 1.3 Score
    res_s = jev.score(
        state=state_refund,
        question="How urgent is the request?",
        levels=[
            "Level 0: No time limit is mentioned; general timeline.",
            "Level 1: A deadline is mentioned but is not immediate.",
            "Level 2: The request needs an immediate response within the day."
        ]
    )
    print(f"\n3. [Score] 工单紧急度评分:")
    print(f"   ✓ 判定等级: {res_s['level_description']}")
    print(f"   ✓ 期望紧急度分数: {res_s['expected_score']:.3f} / 2.0 (时延: {res_s['wall_ms']}ms)")

    # -------------------------------------------------------------
    # 测试集 2：50x50 迷宫探索决策 (Maze Navigation)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【测试集 2：50x50 Maze 迷宫探索单步决策】")
    print("-" * 75)
    state_maze = {
        "position": [12, 18],
        "local_grid_3x3": [
            ["#", " ", "#"],
            ["#", "P", " "],
            ["#", "#", "#"]
        ],
        "open_directions": ["North", "East"],
        "blocked_directions": ["West", "South"],
        "target_exit_direction": "East-North-East",
        "visited_cells_nearby": ["North"]
    }
    print("▶ 迷宫状态 (State)：玩家处于 [12, 18]，西面和南面是实心墙壁(#)，北面已走过，东面未探索且朝向出口。")

    res_maze = jev.decide_choice(
        state=state_maze,
        question="Which movement direction maximizes forward exploration toward the exit while avoiding backtracking?",
        options={
            "move_east": "Move East into the unvisited open corridor leading towards the exit.",
            "move_north": "Move North into the previously visited open corridor.",
            "move_west": "Attempt to move West into the solid wall.",
            "move_south": "Attempt to move South into the solid wall."
        }
    )
    print(f"\n1. [Maze 动作选择] 推荐行动:")
    print(f"   ✓ 选择动作: 【{res_maze['best_action']}】 ({res_maze['description']})")
    print(f"   ✓ 命中概率: {res_maze['probability']*100:.2f}% (领先次选 Margin: {res_maze['margin']*100:.2f}%, 时延: {res_maze['wall_ms']}ms)")

    # -------------------------------------------------------------
    # 测试集 3：贪吃蛇即时避障与觅食 (Snake Game Decisions)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【测试集 3：Snake 贪吃蛇局部避碰与寻路】")
    print("-" * 75)
    state_snake = {
        "snake_head": [5, 5],
        "snake_direction": "Right",
        "snake_length": 14,
        "food_position": [5, 8],
        "immediate_obstacles": {
            "straight (Right)": "Free corridor leading directly to food in 3 steps.",
            "turn_left (Up)": "Body segment obstacle (self-collision hazard).",
            "turn_right (Down)": "Wall boundary (wall collision hazard)."
        }
    }
    print("▶ 贪吃蛇状态 (State)：蛇头向右，正前方直行通往食物；左侧是自身蛇身，右侧是墙壁。")

    res_snake = jev.decide_choice(
        state=state_snake,
        question="Which action should the snake execute to safely advance towards food without dying?",
        options={
            "go_straight": "Continue straight Right along the open path towards food.",
            "turn_left": "Turn left Up into the snake body segment.",
            "turn_right": "Turn right Down into the outer boundary wall."
        }
    )
    print(f"\n1. [Snake 存活动作]:")
    print(f"   ✓ 选择动作: 【{res_snake['best_action']}】 ({res_snake['description']})")
    print(f"   ✓ 命中概率: {res_snake['probability']*100:.2f}% (领先次选 Margin: {res_snake['margin']*100:.2f}%, 时延: {res_snake['wall_ms']}ms)")

    # -------------------------------------------------------------
    # 测试集 4：ViZDoom 射击场景 (Basic & Predict Position 预判发射)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【测试集 4：ViZDoom 射击决策 (瞄准开火 & 移动预判)】")
    print("-" * 75)

    # 4.1 Basic: 瞄准就开火
    state_vizdoom_basic = {
        "game": "ViZDoom Basic",
        "player_health": 100,
        "ammo": 15,
        "crosshair_status": "Lined up exactly with enemy monster center mass",
        "enemy_distance_m": 8.5,
        "enemy_status": "Stationary target visible in reticle"
    }
    print("▶ ViZDoom Basic 状态：十字准心已完美对准站立怪物中心，弹药充足。")
    res_basic = jev.decide_choice(
        state=state_vizdoom_basic,
        question="What is the optimal combat action given the reticle is locked onto the stationary enemy?",
        options={
            "attack": "Fire weapon immediately to eliminate the target.",
            "turn_left": "Turn left away from the target.",
            "turn_right": "Turn right away from the target.",
            "move_backward": "Back away without firing."
        }
    )
    print(f"1. [ViZDoom Basic 动作]: 【{res_basic['best_action']}】 (概率: {res_basic['probability']*100:.2f}%, Margin: {res_basic['margin']*100:.2f}%)")

    # 4.2 Predict Position: 移动目标发射时机
    state_vizdoom_predict = {
        "game": "ViZDoom Predict Position",
        "weapon": "Rocket Launcher (projectile speed: 35 units/s, travel time: 0.6s)",
        "enemy_motion": "Moving rapidly from left to right across field of view",
        "current_crosshair": "Aimed directly at enemy current position (without lead offset)",
        "tactical_assessment": "Firing now will result in rocket passing behind the moving enemy due to travel latency."
    }
    print("\n▶ ViZDoom Predict Position 状态：火箭弹有 0.6s 飞行时间，怪物正在快速横向奔跑，准心当前未打提前量。")
    res_predict = jev.decide_choice(
        state=state_vizdoom_predict,
        question="What is the tactically correct action for leading the shot and eliminating the moving enemy?",
        options={
            "lead_and_aim_ahead": "Turn right to aim ahead of the enemy's projected path to apply lead offset before firing.",
            "fire_directly": "Fire rocket immediately at the enemy's current position despite the 0.6s flight delay.",
            "turn_away": "Turn left in the opposite direction of the enemy's movement.",
            "cease_fire": "Stop moving and wait for ammo to deplete."
        }
    )
    print(f"2. [ViZDoom 移动预判]: 【{res_predict['best_action']}】 ({res_predict['description']})")
    print(f"   ✓ 命中概率: {res_predict['probability']*100:.2f}% (领先次选 Margin: {res_predict['margin']*100:.2f}%, 时延: {res_predict['wall_ms']}ms)")

    print("\n" + "=" * 75)
    print("  🏁 全套环境与游戏实操场景测试完成！")
    print("=" * 75)

if __name__ == "__main__":
    run_game_replay()
