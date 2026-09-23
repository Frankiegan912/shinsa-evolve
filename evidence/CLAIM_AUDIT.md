# ShinsaEvolve 证据冻结报告

本报告只描述源代码版本 `623ffcd760f63b28cdd9210f109751b58634f92e`
下现存的本地实验归档。它不把候选 episode、同一 archive 中的候选或 audit
seeds 当作独立外层实验重复。

## 复核方法

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/verify_evidence.py
```

验证脚本执行以下只读操作：

- 解析六个正式 run 的 `archive.jsonl`；
- 检查完成、失败、recipe 和 parameter 数量；
- 复算 gap、mode-specific pick、best available 和 matched-budget pick；
- 使用受限 AST 解释器读取 `schedule(0)`，不导入或执行生成的 recipe；
- 汇总 mutation attempt 和 seed-recipe retrain；
- 对每个 run 的 archive、配置、recipes 和 parameters 计算确定性树哈希。

机器可读结果见 `evidence/evidence_manifest.json`。当前核验状态为 `pass`。

## 可由现有归档复算的主张

| 主张 | 结果 | 证据状态 |
|---|---:|---|
| Breakout audited gap mean / positive | 0.3729 / 36 of 60 | 可复算 |
| Breakout claimed gap mean / positive | 1.1029 / 18 of 20 | 可复算 |
| Lunar audited gap mean / positive | 140.7893 / 38 of 40 | 可复算 |
| Lunar claimed gap mean / positive | 241.2742 / 15 of 15 | 可复算 |
| Breakout matched audited vs claimed pick | 4.9375 vs 4.3125, N=20 | 可复算 |
| Lunar matched audited vs claimed pick | -62.4796 vs -118.8142, N=15 | 可复算 |
| mutation first-attempt acceptance | 129 of 131 | 可复算 |
| mean recorded CLI latency | 95.77 seconds | 可复算 |
| Breakout seed-recipe range | 3.53125 to 5.0, four trainings | 可复算 |
| Lunar seed-recipe range | -236.5130 to -128.4131, four trainings | 可复算 |

论文中的 schedule(0) 均值也可从保存的 recipe 静态复算。这里的“last 10”指
按候选编号排序后的最后十个完成 recipe，而不是独立运行。

## 只能部分支持的主张

- 审计实现确实生成 OS-entropy seeds，并只把 policy parameters 传给 raw-reward
  evaluator；但 archive 没有保存 seeds 或 per-episode returns，历史执行无法逐 episode
  重放。
- Lunar claimed champion 的 recipe 和 schedule 已保存，其 shaping 看起来温和；但
  “measurement gaming rather than reward hacking” 仍是机制解释，不是已隔离的因果实验。
- 单次 matched-budget archive 中 audited-mode pick 更好；这支持对现有轨迹的描述，
  不支持 “audit restores selection quality” 的一般结论。
- Lunar audited champion 超过四次 seed-recipe 训练的观测范围；champion recipe 没有
  独立重训，因此不能据此声称 recipe 层面的稳定改进。
- 保存的 mutation logs 证明 131 个完成候选经过 CLI mutation，但没有记录精确模型
  ID、CLI 版本或计费信息，无法验证论文中的具体 Claude Sonnet 版本。

## 当前无法从归档验证的主张

- 论文机制段的 `+282.2` replay、`-121.7` fresh-audit mean、4.4% leg-contact 和
  `-2.1` shaping contribution 没有独立结果文件或对应 seeds。
- audit seed overfitting 没有通过改变 seed reuse 的消融实验隔离。
- clean-room 来源声明不能仅由当前文件内容证明；扫描只表明当前树和现有 Git 历史
  未发现预先定义的受限来源关键词。
- 每个 environment/mode 只有一次独立 outer-loop run，因此不能从当前数据估计
  selection effect 的运行间方差。

## 已确认的不一致与实现限制

- 论文所写 LunarLander MLP 参数量 1,508 与代码不符；按 `(8, 32, 32, 4)` 应为
  1,476。
- episode budget 是代际边界上的硬约束；wall-clock budget 只在每代后检查，不能抢占
  超时的一代。
- 静态检查禁止裸 `open` 等调用，但不能阻止所有通过允许模块属性触发的 I/O；
  “rejects IO” 是过强表述。
- resume 不持久化 outer-loop RNG 状态，也不会拒绝不兼容的新配置，因此不能保证与
  不间断运行等价。
- Breakout audited run 包含三个归档的 mutation failures，并曾发生一个未归档的训练期
  `NameError`；恢复后同一 candidate ID 被重新使用，原失败 recipe 已被覆盖。

## 结论边界

当前证据足以支持：在四条已归档搜索轨迹中，claimed score 与独立 raw-reward audit
出现明显偏离，并导致 archive 内误排序。

当前证据不足以支持：该效应系统性发生、audit 一般性恢复选择质量、机制已经被因果
隔离，或演化 recipe 能稳定优于 baseline。
