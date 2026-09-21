"""Paired same-case comparison; does not change training or select checkpoints."""
import collections
import hashlib
import json
from pathlib import Path
import random
ROOT=Path(__file__).parent

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def correct(r):return max(range(len(r['probs'])),key=r['probs'].__getitem__)==max(range(len(r['target'])),key=r['target'].__getitem__)

def bootstrap(a,b):
    left={r['id']:r for r in a};right={r['id']:r for r in b};assert left.keys()==right.keys()
    groups=collections.defaultdict(list)
    for key,r in left.items():
        assert r['target']==right[key]['target'];groups[r['case_id']].append(int(correct(r))-int(correct(right[key])))
    values=[sum(v)/len(v) for v in groups.values()];rng=random.Random(20260921)
    draws=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(2000))
    return {'accuracy_delta':sum(values)/len(values),'case_bootstrap_95_interval':[draws[49],draws[1949]],'cases':len(values)}

def main():
    ours=read('agentjev_v1/report.json');laya=read('laya_test_report.json')
    after=read('agentjev_v1/test_calibrated_predictions.json');before=read('agentjev_v1/phase4_test_predictions.json');other=read('laya_test_predictions.json')
    comparisons={'trained_vs_phase4':bootstrap(after,before),'trained_vs_laya':bootstrap(after,other)}
    payload={'agentjev':ours,'laya':laya,'paired_comparisons':comparisons}
    (ROOT/'comparison.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    rows=[('AgentJev 原 Phase 4',ours['phase4']['overall']),('AgentJev 本轮训练',ours['trained']['overall']),
          ('AgentJev 本轮训练＋独立校准',ours['trained_calibrated']['overall']),('Laya typed-decisions 固定权重',laya['metrics']['overall'])]
    lines=['# AgentJev 与 Laya 同题对照','',
        '数据先在本地下载，再经 SFTP 上传服务器；10 个下载文件 SHA256 全部匹配。模型、数据和源码版本见 download_manifest.json。','',
        '使用官方测试集 400 个案例 / 2,000 个决策。AgentJev 按协议仅从官方训练集划出 960 个训练、120 个开发、120 个校准案例；同一案例的所有问题属于同一分割，未向模型输入 factors、gold 或案例 ID。',
        'Laya 使用公开的任务专用 checkpoint，而非较弱的通用基础 checkpoint；该权重使用过完整官方训练集，因此这里不是等训练预算的架构比较。','',
        '| 模型 | 准确率 | 软标签交叉熵↓ | Brier（按候选求和）↓ | ECE↓ | Score 期望误差↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in rows:lines.append(f"| {name} | {m['accuracy']:.2%} | {m['soft_cross_entropy']:.4f} | {m['brier_sum']:.4f} | {m['ece_10_bins_vs_gold_argmax']:.4f} | {m['score_expectation_mae']:.4f} |")
    lines+=['',f"选中第 {ours['selection']['best_step']} 步，选择依据为开发集软标签交叉熵；选中权重后才打开测试集评测。",'']
    for key,title in [('trained_vs_phase4','相对原 AgentJev'),('trained_vs_laya','相对 Laya')]:
        c=comparisons[key];lo,hi=c['case_bootstrap_95_interval']
        lines.append(f"{title}：准确率差 {c['accuracy_delta']*100:+.2f} 个百分点，按案例重采样 95% 区间 [{lo*100:+.2f}, {hi*100:+.2f}]。")
    lines+=['','## 指标解释','',
        '准确率表示与公开教师分布最大概率选项的一致率；Brier 为预测分布与教师分布的平方差，按候选求和再按问题平均。ECE 使用最大概率和该一致率，10 个等宽区间。Score 误差比较两个分布的期望等级。不同项目公布的 Brier、soft accuracy 定义可能不同，因此以本次统一重算为准。',
        '数据包含合成场景和教师概率标签。这一结果验证该基准上的决策学习，不能替代真实 Coding Agent 成功率，也不能推广为全面超过 Jev。',
        '训练采用软交叉熵＋0.1 Brier，先建立可复现的直接监督基线；没有将这一轮称为 RLCD 训练。','',
        '完整输出：agentjev_v1/report.json、laya_test_report.json、comparison.json，以及逐题概率文件。原权重和原推理服务未覆盖。','']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
