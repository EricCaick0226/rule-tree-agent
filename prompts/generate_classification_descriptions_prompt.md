# 任务

基于输入 `input_payload.rows` 中每条分类行的检索上下文，生成候选 `分类说明`。只返回 JSON object。

# 必须遵守

- 只能使用 `retrieved_contexts[].text` 中的信息，不得使用文档外知识。
- `proposed_description` 必须是解释性说明，不要复制 `current_description`、叶子节点名或 `数据范围及示例`。
- 分类说明只解释分类项是什么、面向什么对象或业务、通常包含哪类数据。
- 可以参考 `数据加工程度`、`影响对象`、`影响程度`、`数据级别` 理解上下文，但不要写推荐分级，不要写“定级为几级”，不要写“依据文档定级为……”。
- 不要把分类说明写成分级理由；分级因素只在必要时用来辅助说明数据性质。
- 如果检索上下文不足以解释该分类项，返回 `证据不足，无法从当前文档确定`。
- `description_source` 只能是 `summarized` 或 `insufficient`。
- `description_evidence_quote` 必须来自 `retrieved_contexts[].text` 的原文片段；如果证据不足则为空字符串。
- 生成的说明属于候选结果，`needs_review` 必须为 `true`。
- 输出字段名和枚举值保持英文，不要翻译 JSON key。

# 输出格式

返回如下 JSON object：

```json
{
  "description_candidates": [
    {
      "row_id": "",
      "proposed_description": "",
      "description_source": "summarized",
      "description_evidence_quote": "",
      "needs_review": true,
      "review_reason": "基于检索上下文总结生成，需要人工确认。"
    }
  ]
}
```
