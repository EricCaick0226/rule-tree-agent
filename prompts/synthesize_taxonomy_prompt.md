# 分类树合成 Prompt

任务：基于 `evidence_claims` 和 `concept_profiles` 构建候选分类树。

规则：
- 只能使用输入中提供的 `evidence_claims` 和 `concept_profiles`。
- 不得编造根分类。
- 不得编造分类层级。
- 不得强制生成固定深度的树。
- 本步骤不得生成分级结果或匹配规则。
- 每一个节点以及每一条父子关系都必须返回对应的 `evidence_claim_ids`。
- 优先使用 `support_level` 为 `explicit` 或 `structural` 的 claim。
- 如果节点或父子关系只由 `inferred`、`weak` 或 `ocr` claim 支持，必须设置 `needs_review = true`。
- 如果层级证据较弱，必须设置 `needs_review = true`。
- 如果证据不足，返回空的节点列表，或返回标记为 `insufficient_evidence` 的项目。

只输出 JSON object，不要输出 Markdown、解释文字或额外说明。
