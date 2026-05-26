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
- 每一个 claim 都必须返回对应的 `evidence_chunk_ids`。
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

只输出 JSON object，不要输出 Markdown、解释文字或额外说明。
