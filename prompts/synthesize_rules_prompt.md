# 规则生成 Prompt

任务：为已有候选节点生成基于证据的匹配规则。

规则：
- 只能使用输入中提供的节点和 `evidence_claims`。
- 不得编造关键词。
- 不得编造正则表达式。
- 除非字段示例明确出现在证据中，否则不得使用字段示例。
- `conditions` 和 `negative_conditions` 必须来自证据 claim。
- 优先使用 `support_level` 为 `explicit` 或 `structural` 的 claim。
- 如果规则只由 `inferred`、`weak` 或 `ocr` claim 支持，必须设置 `needs_review = true`。
- 只有在存在明确排除证据时，才能生成负向规则。
- 每一条规则都必须返回对应的 `evidence_claim_ids`。
- 如果证据不足，必须输出 `insufficient_evidence` 并设置 `needs_review = true`。

只输出 JSON object，不要输出 Markdown、解释文字或额外说明。
