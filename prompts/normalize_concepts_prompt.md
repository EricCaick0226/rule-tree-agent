# 概念归一 Prompt

任务：基于 `evidence_claims` 构建概念画像。

规则：
- 只能使用输入中提供的 `evidence_claims`。
- 不得编造分类类别。
- 不得编造分级等级。
- 不得编造分类层级。
- 不得编造节点描述。
- 不得编造匹配规则。
- 概念名称、别名、定义、包含项和排除项都必须能追溯到 claim ID。
- 如果证据不足，必须输出 `needs_review = true`。
- 每一个概念画像都必须返回对应的 `related_claim_ids`。
- 必须区分分类、分级、规则和证据，不要把它们混在一起。

只输出 JSON object，不要输出 Markdown、解释文字或额外说明。
