# 文档证据驱动规则树生成 Agent 架构说明

## 项目目标

本项目用于从已有文档中生成候选分类与分级规则树。它不是企业标准制定工具，而是一个人机协同的辅助 Agent：AI 只提出候选结果，人类负责审核、修订和批准。

## 为什么不能硬编码分类

不同企业、不同部门、不同制度文档中的分类原则可能完全不同。如果在代码中写死分类名称、等级名称、风险规则或行业示例，系统就会把外部先验误当成文档事实，导致输出无法审计，也无法证明来自输入材料。

因此，所有业务内容必须来自输入文档，包括：

- 分类节点名称
- 分类层级
- 分级名称
- 分级定义
- 节点说明
- 匹配规则中的关键词、短语、示例和排除条件

如果文档没有提供足够证据，系统必须输出证据不足、需要人工复核或无法从当前文档确定。

## 核心概念区别

- 证据：来自原始文档的文本片段，是所有生成内容的依据。
- 证据 claim：从证据中抽取出的原子事实，必须包含原文短引文、证据强度、证据引用和需要复核时的原因。
- 文档块信号：判断 chunk 是否承载分类明细、分级定义或其他支持性证据。
- classification_rows：默认链路的主事实表，每行包含 path_levels、推荐等级、说明、证据引用和复核标记。
- 分级定义：文档中明确提供的等级名称、定义和属性；没有定义就不能默认创建。
- 派生分类树：从 classification_rows 的 path_levels 确定性投影出来的审阅视图，不是 LLM 直接生成事实。
- 复核：对低置信度、弱证据、缺失证据或无法判断的内容进行人工确认。

## 工作流

新版默认链路以 classification_rows 为主事实表。分类树不是 LLM 直接生成结果，而是从每一行的 path_levels 确定性派生。旧的 tree-first steps 保留在仓库中作为 legacy 对照，但默认 pipeline 不再运行。

1. 解析文档：默认 row-first MVP 读取 `.md` 和 `.txt` 输入；PDF/OCR 解析代码仍保留在仓库中，但不属于默认链路。
2. 切分文档：按标题、编号、空行和列表块形成带 source span 的 chunk。
3. 构建证据索引：记录 chunk、文档、章节之间的关系。
4. 抽取 evidence claims：必须调用 LLM，按 chunk 批量抽取文档事实，不生成树。
5. 判断文档块信号：必须调用 LLM，识别哪些块承载分类明细、分级定义或其他支持性证据。
6. 抽取 classification_rows：必须调用 LLM，只从证据块中抽取路径、推荐等级、说明和证据引用。
7. 抽取分级定义：必须调用 LLM，只抽取文档明确提供的等级名称、定义和属性。
8. 归一 classification_rows：确定性去重、规范空白、标记弱证据，不新增文档外分类。
9. 校验行级证据：检查每行路径、说明、等级和引用是否可追溯。
10. 派生分类树：从每行 `path_levels` 确定性生成树视图。
11. 导出结果：生成候选表、派生规则树、人工复核报告和原始 LLM trace 文件。

## 模块职责

- `src/core/agent_state.py`：定义文档、证据、文档块信号、classification_rows、分级定义、派生节点、校验问题和整体状态。
- `src/io/document_parser.py`：负责默认 row-first 的 txt/md 解析入口与 chunk 切分，不解释业务含义。
- `src/io/pdf_document_parser.py`：legacy/non-default PDF 文本层抽取、按需调用 macOS Vision OCR，并生成带页码和来源方式的原文转写。
- `src/io/evidence_store.py`：创建证据引用，提供简单本地关键词搜索。
- `src/io/evidence_index.py`：建立 chunk 与文档、章节的索引。
- `src/steps/evidence_claim_extractor.py`：LLM claim 抽取步骤，只产生证据事实。
- `src/steps/block_classifier.py`：LLM 文档块信号识别步骤。
- `src/steps/classification_row_extractor.py`：LLM 分类分级明细行抽取步骤。
- `src/steps/grade_definition_extractor.py`：LLM 分级定义抽取步骤。
- `src/steps/classification_row_normalizer.py`：classification_rows 确定性归一与去重步骤。
- `src/steps/tree_projector.py`：从 classification_rows 的 path_levels 确定性派生分类树。
- `src/steps/concept_normalizer.py`、`src/steps/dimension_analyzer.py`、`src/steps/taxonomy_synthesizer.py`、`src/steps/node_describer.py`、`src/steps/grading_analyzer.py`、`src/steps/rule_synthesizer.py`：legacy tree-first 对照步骤，默认 pipeline 不运行。
- `src/validation/row_grounding_validator.py`：默认 row-first 校验器，严格检查分类分级明细行、分级和派生树是否有证据。
- `src/validation/grounding_validator.py`：legacy tree-first 对照校验器，默认 pipeline 不运行。
- `src/llm/client.py`：OpenAI-compatible LLM 客户端，默认模型为 `your-model-name`。
- `src/llm/task_utils.py`：加载 prompt 文件，统一 LLM JSON 调用、顶层 schema 校验和一次重试。
- `src/output/exporter.py`：导出候选表、派生树、复核报告和原始 LLM trace。
- `src/pipeline/agent_executor.py`：串联整个 MVP 流水线。
- `src/agent_demo.py`：命令行演示入口。

## 数据结构

系统状态将内容分为主要事实表和派生视图：

- Evidence
- Block signals
- Classification rows
- Grade definitions
- Derived tree nodes
- Validation issues
- Human review status

`classification_rows` 是默认链路的主事实表；树节点只是从行路径派生出来的审阅视图。这种分离可以防止把生成内容误当成事实，也方便审计每个候选结论的来源。

## MVP 边界

- 默认 row-first MVP 只支持 Markdown 和纯文本输入。
- 底层 PDF/OCR 解析代码仍在仓库中作为 legacy/non-default 基础设施，但默认 row-first agent 不接受 PDF/OCR 输入；`agent_demo` / `run_agent` 会拒绝非 `.txt`/`.md` 文件。
- 后续若恢复 PDF/OCR 到默认链路，需要单独设计解析、证据定位、OCR 质量复核和 row grounding 策略。
- 只使用本地规则、关键词和轻量模糊匹配。
- 必须接入 OpenAI-compatible LLM，LLM 调用失败即任务失败。
- 运行时 prompt 来自 `prompts/`，每个 LLM 阶段只允许处理当前阶段职责。
- LLM 原始回复不写入主 `rule_tree.json`，而是单独写入 `outputs/traces/`。
- 不使用向量数据库。
- 不处理复杂表格、版面重建、低质量扫描修复或跨文档冲突消解。
- 输出是候选结果，不是最终标准。

## 未来版本想法

- 继续增加更细的嵌套 JSON schema 校验和跨阶段契约检查。
- 增加表格解析和单元格级证据引用。
- 支持人工反馈后重新生成。
- 支持多版本文档对比与冲突标注。
- 支持更丰富的导出格式和审批流集成。
- 增加更严格的可追溯性测试。
