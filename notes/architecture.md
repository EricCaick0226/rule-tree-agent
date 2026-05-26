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
- 概念：从标题、列表、定义句、结构化语句中抽取出的候选术语。
- 分类维度：文档中说明的分类依据，例如“按照某某分类”中的某某；如果只是推断，必须标记复核。
- 分类节点：规则树中的节点，名称和层级必须有证据。
- 分级：文档中定义的等级名称、定义和条件；没有定义就不能默认创建。
- 规则：用于辅助匹配节点的关键词、短语、上下文或排除条件；只能来自证据。
- 复核：对低置信度、弱证据、缺失证据或无法判断的内容进行人工确认。

## 工作流

1. 解析文档：读取 `.md`、`.txt` 和 `.pdf`；PDF 先用文本层抽取，必要时在显式开启 OCR 后用 macOS Vision 对无文本页进行 OCR。
2. 切分文档：按标题、编号、空行和列表块形成 chunk。
3. 构建证据索引：记录 chunk、文档、章节之间的关系。
4. 抽取 evidence claims：必须调用 LLM，按 chunk 批量抽取文档事实，不生成树。
5. 归一概念画像：必须调用 LLM，把 claims 组织成概念、定义、包含项、排除项。
6. 发现分类维度：必须调用 LLM，只判断文档支持的分类依据。
7. 合成候选分类树：必须调用 LLM，只生成节点与父子关系；如果证据不足，允许输出空候选树并进入复核报告。
8. 生成节点描述：必须调用 LLM，只基于 claims 写定义和范围。
9. 分析分级方案：必须调用 LLM，只抽取等级定义和节点映射。
10. 生成匹配规则：必须调用 LLM，只使用 claims 中的术语、短语和排除关系。
11. 校验证据：检查节点、描述、等级、规则和维度是否可追溯。
12. 导出结果：生成 JSON、候选规则树 Markdown、人工复核报告和原始 LLM trace 文件。

## 模块职责

- `src/core/agent_state.py`：定义文档、证据、概念、维度、节点、等级、规则、校验问题和整体状态。
- `src/io/document_parser.py`：只负责普通文本解析入口与 chunk 切分，不解释业务含义。
- `src/io/pdf_document_parser.py`：只负责 PDF 文本层抽取、按需调用 macOS Vision OCR，并生成带页码和来源方式的原文转写。
- `src/io/evidence_store.py`：创建证据引用，提供简单本地关键词搜索。
- `src/io/evidence_index.py`：建立 chunk 与文档、章节的索引。
- `src/steps/evidence_claim_extractor.py`：LLM claim 抽取步骤，只产生证据事实。
- `src/steps/concept_normalizer.py`：LLM 概念画像步骤。
- `src/steps/dimension_analyzer.py`：LLM 分类维度分析步骤。
- `src/steps/taxonomy_synthesizer.py`：LLM 候选分类树合成步骤。
- `src/steps/node_describer.py`：LLM 节点描述生成步骤。
- `src/steps/grading_analyzer.py`：LLM 分级方案与节点分级分析步骤。
- `src/steps/rule_synthesizer.py`：LLM 匹配规则生成步骤。
- `src/validation/grounding_validator.py`：严格检查所有生成内容是否有证据。
- `src/llm/client.py`：OpenAI-compatible LLM 客户端，默认模型为 `your-model-name`。
- `src/llm/task_utils.py`：加载 prompt 文件，统一 LLM JSON 调用、顶层 schema 校验和一次重试。
- `src/output/exporter.py`：导出结构化结果、候选树、复核报告和原始 LLM trace。
- `src/pipeline/agent_executor.py`：串联整个 MVP 流水线。
- `src/agent_demo.py`：命令行演示入口。

## 数据结构

系统状态将内容分为九类：

- Evidence
- Concepts
- Classification dimensions
- Tree nodes
- Grading scheme
- Node descriptions
- Matching rules
- Validation issues
- Human review status

这种分离可以防止把生成内容误当成事实，也方便审计每个候选结论的来源。

## MVP 边界

- 支持 Markdown、纯文本和 PDF；OCR 必须显式开启，当前 OCR 后端依赖 macOS Vision，且 OCR 证据默认需要人工复核。
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
