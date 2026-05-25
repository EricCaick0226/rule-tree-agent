# 分类维度发现 Prompt

任务：基于 `evidence_claims` 和 `concept_profiles` 发现文档支持的分类维度。

规则：
- 只能使用输入中提供的 `evidence_claims` 和 `concept_profiles`。
- 不得编造分类类别。
- 不得编造分级等级。
- 不得编造分类层级。
- 不得编造节点描述。
- 不得编造匹配规则。
- 优先使用明确表达分类原则、分类依据、分类维度或划分方式的证据 claim。
- 如果不存在可靠分类维度，必须设置 `selected_dimension_name = null`。
- 每一个分类维度都必须返回对应的 `evidence_claim_ids`。
- 如果证据较弱，必须输出 `needs_review = true`。
- 必须区分分类、分级、规则和证据，不要把它们混在一起。

只输出 JSON object，不要输出 Markdown、解释文字或额外说明。
