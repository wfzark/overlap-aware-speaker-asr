# Team Contributions

## 分工说明

| 成员 | 主要贡献 | 模块 |
| --- | --- | --- |
| 成员1 | 数据准备、音频切分、Whisper ASR 基准测试 | `src/whisper_transcribe.py`, `src/separate_speakers.py` |
| 成员2 | Adaptive Router v1/v2 设计与实现 | `src/adaptive_router_v2.py`, `src/risk_aware_selector.py` |
| 成员3 | 后处理（去重/清洗）、CER 评估管线 | `src/postprocess.py`, `src/evaluate_cer.py` |
| 成员4 | LLM Repair Loop + RAG 整合 | `src/llm_repair_loop.py`, `src/rag_repair.py` |
| 成员5 | Synthetic 数据生成与泛化实验 | `src/synthetic_*.py` |
| 成员6 | Streamlit Demo、可视化、报告撰写 | `src/demo_app.py`, `src/router_feature_importance.py` |

## Commit 规范

- feat: 新功能
- fix: 修复
- docs: 文档
- refactor: 重构
- eval: 评估实验

## 代码审查

所有 PR 需至少一人 review 后合并。
