"""
AgentJev 编程与代码编写实战场景评测
===================================
通过 4 个高频、硬核的软件开发决策场景，实测 AgentJev 0.6B (端口 8149) 的判断力：
1. 代码安全审查 (SQL 注入缺陷与合并门控)
2. 真实 Bug 根因定位与修复策略 (递归死循环与图环路)
3. 性能架构决策 (高并发订单瓶颈选型)
4. 重构风险评估 (大型重构与回归风险评分)
"""

import json
from agentjev_client import AgentJev

def run_coding_benchmarks():
    jev = AgentJev("http://127.0.0.1:8149")
    print("="*75)
    print("  🚀 AgentJev 0.6B 软件工程与代码编写决策能力实战评测")
    print("="*75)

    # -------------------------------------------------------------
    # 场景 1：代码安全审查（Code Review / SQL 注入缺陷识别）
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【场景 1：PR 代码安全审查与合并门控】")
    print("-" * 75)
    state_cr = {
        "pr_title": "Fix: Add search filter to user management endpoint",
        "author": "junior_dev",
        "changed_files": ["backend/services/user_service.py"],
        "diff": (
            "@@ -28,5 +28,8 @@ class UserService:\n"
            "     def search_users(self, keyword: str, tenant_id: str):\n"
            "-        return self.db.query(User).filter_by(tenant_id=tenant_id).all()\n"
            "+        # Concatenate keyword directly into raw query for flexible search\n"
            "+        raw_sql = f\"SELECT * FROM users WHERE tenant_id = '{tenant_id}' AND (username LIKE '%{keyword}%' OR email = '{keyword}')\"\n"
            "+        return self.db.execute(raw_sql).fetchall()\n"
        ),
        "ci_status": "Passed (unit tests passed with mocked db)"
    }
    print("▶ 提交的代码 Diff：")
    print("  开发者将 ORM 查询改成 f-string 拼接的原始 SQL: f\"SELECT * FROM ... WHERE username LIKE '%{keyword}%'\"")

    # Q1: Boolean 漏洞判定
    res_vuln = jev.decide_boolean(
        state=state_cr,
        question="Does this code change introduce a critical security vulnerability (such as SQL Injection)?",
        criteria={
            "true": "The code allows untrusted user input directly into executable SQL queries.",
            "false": "The code uses safe parameterized queries or sanitization."
        }
    )
    print(f"\n1. [Boolean] 是否引入重大安全漏洞 (SQL 注入)?")
    print(f"   ✓ 模型判定: {'【存在严重安全漏洞】' if res_vuln['decision'] else '【安全】'}")
    print(f"   ✓ 漏洞存在概率: {res_vuln['prob_true']*100:.2f}% | 安全概率: {res_vuln['prob_false']*100:.2f}% (耗时: {res_vuln['wall_ms']}ms)")

    # Q2: Choice 评审动作
    res_action = jev.decide_choice(
        state=state_cr,
        question="What should the code reviewer do with this Pull Request?",
        options={
            "approve_and_merge": "Approve and merge immediately since unit tests passed.",
            "request_changes_parameterize": "Request changes: require parameterized query or ORM filter to eliminate SQL injection vulnerability.",
            "rewrite_in_rust": "Close PR and demand rewriting the whole service in Rust.",
            "ignore_security": "Merge to master and monitor in production."
        }
    )
    print(f"\n2. [Choice] Reviewer 应该做出什么决策?")
    print(f"   ✓ 推荐动作: 【{res_action['best_action']}】")
    print(f"   ✓ 决策说明: {res_action['description']}")
    print(f"   ✓ 置信度: {res_action['probability']*100:.2f}% (领先次选 Margin: {res_action['margin']*100:.2f}%, 耗时: {res_action['wall_ms']}ms)")

    # -------------------------------------------------------------
    # 场景 2：故障排查与 Bug 修复方案决断 (Bug Fixing Strategy)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【场景 2：生产递归死循环 Bug 修复策略抉择】")
    print("-" * 75)
    state_bug = {
        "exception": "RecursionError: maximum recursion depth exceeded while calling a Python object",
        "stack_trace": (
            "File '/app/graph_traversal.py', line 45, in traverse_nodes\n"
            "    return [traverse_nodes(child) for child in node.children]\n"
            "    ... repeated 998 times ...\n"
        ),
        "code_snippet": (
            "def traverse_nodes(node):\n"
            "    result = [node.name]\n"
            "    for child in node.children:\n"
            "        result.extend(traverse_nodes(child))\n"
            "    return result\n"
        ),
        "incident_cause": "A newly introduced cyclical reference in the dependency tree graph caused infinite recursion."
    }
    print("▶ 报错与代码：traverse_nodes 深度遍历图节点时由于数据存在环路（Cycle），触发 RecursionError 爆栈崩溃。")

    res_fix = jev.decide_choice(
        state=state_bug,
        question="Which bug fix implementation is the correct and robust solution for this RecursionError?",
        options={
            "add_visited_set": "Pass a 'visited' set (or convert to iterative BFS/DFS with visited tracking) to detect and prevent cycles.",
            "increase_recursion_limit": "Simply call sys.setrecursionlimit(1000000) at application startup to suppress the error.",
            "catch_and_ignore": "Wrap the recursive call in a try/except RecursionError block and return an empty list.",
            "restart_pod": "Do not modify the code; restart the pod whenever it crashes."
        }
    )
    print(f"\n1. [Choice] 哪一个才是正确健壮的修复代码方案?")
    print(f"   ✓ 推荐修复方案: 【{res_fix['best_action']}】")
    print(f"   ✓ 方案描述: {res_fix['description']}")
    print(f"   ✓ 模型判定概率: {res_fix['probability']*100:.2f}% (领先次选 Margin: {res_fix['margin']*100:.2f}%, 耗时: {res_fix['wall_ms']}ms)")
    print(f"   ✓ 候选项概率分布: {json.dumps({k: round(v*100, 2) for k, v in res_fix['distribution'].items()})}")

    # -------------------------------------------------------------
    # 场景 3：高并发瓶颈技术选型 (Architecture Decision)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【场景 3：高并发写入瓶颈架构重构选型】")
    print("-" * 75)
    state_arch = {
        "system": "Flash Sale Checkout Service (秒杀下单系统)",
        "current_architecture": "Monolithic REST API directly executing synchronous ACID transactions on single PostgreSQL primary",
        "performance_bottleneck": (
            "During peak flash sale, TPS spikes from 200 to 8,000. PostgreSQL CPU reaches 100% with massive row-level lock contention "
            "on the 'inventory' table. Over 70% of HTTP requests timeout or return 504 Gateway Timeout."
        )
    }
    print("▶ 系统瓶颈：秒杀峰值 TPS 飙升至 8000，单体同步直写数据库导致库存行锁严重争抢，大量请求超时。")

    res_arch = jev.decide_choice(
        state=state_arch,
        question="Which architectural pattern is the most effective and standard engineering solution to resolve this write bottleneck?",
        options={
            "queue_and_cache_decoupling": "Decouple with Redis Lua script for atomic inventory pre-deduction, and enqueue orders into a message queue (e.g. Kafka/RabbitMQ) for asynchronous database writes.",
            "hardware_upgrade": "Double the RAM and CPU of the existing single PostgreSQL instance without architectural changes.",
            "busy_sleep_retry": "Add while True sleep retry loops inside the web controller to wait for database locks to release.",
            "remove_transactions": "Remove database transactions and consistency checks completely to make writes faster."
        }
    )
    print(f"\n1. [Choice] 推荐的架构解耦重构方案:")
    print(f"   ✓ 最佳架构: 【{res_arch['best_action']}】")
    print(f"   ✓ 架构描述: {res_arch['description']}")
    print(f"   ✓ 置信度: {res_arch['probability']*100:.2f}% (领先次选 Margin: {res_arch['margin']*100:.2f}%, 耗时: {res_arch['wall_ms']}ms)")

    # -------------------------------------------------------------
    # 场景 4：代码重构风险综合定级 (Refactoring Risk Scoring)
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("【场景 4：核心支付状态机重构风险评级】")
    print("-" * 75)
    state_risk = {
        "module": "OrderPaymentStateMachine",
        "change_scope": "Rewrite core payment transition table, changing transaction isolation level from SERIALIZABLE to READ COMMITTED for performance.",
        "test_coverage": "Existing legacy test suite only covers happy paths (62% line coverage), missing edge case double-spend tests.",
        "traffic_impact": "Handles 100% of company revenue transactions."
    }
    print("▶ 重构范围：修改核心支付状态机，降低事务隔离级别提升性能，且缺乏双花等极端异常测试覆盖。")

    res_risk = jev.score(
        state=state_risk,
        question="Rate the deployment and operational risk level of this refactoring change:",
        levels=[
            "Level 0 - Negligible: Internal cosmetic change, low blast radius.",
            "Level 1 - Low: Non-critical feature, well covered by tests.",
            "Level 2 - Moderate: Meaningful change to business logic with adequate fallbacks.",
            "Level 3 - High: Core financial/revenue flow with incomplete test coverage and high regression potential."
        ]
    )
    print(f"\n1. [Score] 重构风险评级结果:")
    print(f"   ✓ 风险定级: {res_risk['level_description']}")
    print(f"   ✓ 期望风险分值: {res_risk['expected_score']:.3f} / 3.0 (耗时: {res_risk['wall_ms']}ms)")
    print(f"   ✓ 分布: {json.dumps({k: round(v*100, 2) for k, v in res_risk['distribution'].items()})}")

    print("\n" + "="*75)
    print("  🏁 编程与代码实战场景评测完毕！")
    print("="*75)

if __name__ == "__main__":
    run_coding_benchmarks()
