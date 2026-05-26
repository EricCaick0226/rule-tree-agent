# 证据 Claim 抽取 Prompt

任务：从文档 chunk 中抽取可追溯的证据 claim。

规则：
- 只能使用输入中提供的 `document_chunks`。
- 不要构建分类树。
- 不得编造分类类别。
- 不得编造分级等级。
- 不得编造分类层级。
- 不得编造节点描述。
- 不得编造匹配规则。
- 如果证据不足，必须输出 `needs_review = true`。
- 如果 chunk 只包含标题、页眉、页脚、目录项或没有实质事实，不要输出 claim；不要为了标题生成 `insufficient_evidence`。
- 输入中的 `chunk_signal` 只是辅助判断信号，不是删除依据；必须仍以 `text` 原文为准。
- 如果 `chunk_signal` 是 `heading_only`、`short_ocr` 或 `possible_noise`，应更谨慎：通常不要输出 claim，除非 `text` 中存在明确事实证据。
- 如果 `chunk_signal` 是 `table_like`，可以基于行列结构抽取 claim，但必须保留能在 `text` 中找到的短原文片段。
- 如果一句话列出多个对象、类别、等级或规则项，必须拆成多个原子 claim，不要把多个对象合并到同一个 `object` 或 `value` 字符串里。
- `subject`、`object`、`value` 和 `evidence_quote` 应尽量短；优先使用原文中的单个术语、短语或一个明确事实。
- 每一个 claim 都必须返回对应的 `evidence_chunk_ids`。
- 每一个 claim 都必须返回 `evidence_quote`，它必须是支持该 claim 的短原文片段，不要改写。
- 每一个 claim 都必须返回 `support_level`，只能使用输入 schema 中允许的英文枚举值。
- 如果 `support_level` 是 `inferred`、`weak` 或 `ocr`，必须设置 `needs_review = true`。
- 如果需要人工复核，必须填写 `review_reason`，说明是 OCR、证据弱、结构推断、冲突或证据不足。
- 如果 chunk 的 `source_method` 是 `ocr`，该 claim 必须设置 `needs_review = true`，因为 OCR 证据可能存在识别误差。
- 必须区分分类、分级、规则和证据，不要把它们混在一起。

允许的 `claim_type` 枚举值如下。这些值是程序识别用的英文标识，不要翻译：
- definition
- inclusion
- exclusion
- hierarchy
- classification_principle
- grade_definition
- grade_mapping
- rule_phrase
- insufficient_evidence

允许的 `support_level` 枚举值如下。这些值是程序识别用的英文标识，不要翻译：
- explicit：原文直接表达定义、包含、排除、等级或映射关系。
- structural：由标题、编号、表格行列、列表层级等文档结构支持。
- inferred：只能从上下文谨慎推断，必须人工复核。
- weak：证据很弱或不完整，必须人工复核。
- ocr：证据来自 OCR，必须人工复核。

只输出 JSON object，不要输出 Markdown、解释文字或额外说明。
