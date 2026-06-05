# 任务

基于输入 `input_payload.rows` 中每条分类行的检索上下文，生成候选 `分类说明`。只返回 JSON object。

# 必须遵守

- 只能使用 `retrieved_contexts[].text` 中的信息，不得使用文档外知识。
- `proposed_description` 必须是解释性说明，不要复制 `current_description`、叶子节点名或 `数据范围及示例`。
- 分类说明只解释分类项是什么、面向什么对象或业务、通常包含哪类数据或属于哪个业务环节；优先改写原文中的分类名和数据范围，不要补充原文未明确支持的用途、目的或效果，也不要把“原始数据/统计数据”等表字段作为说明主体。
- 当 `retrieved_contexts` 同时提供了明确的分类路径、叶子分类名、数据范围或示例时，可以基于这些原文信息改写成候选分类说明，不要求必须存在完整定义句。
- 可以参考 `数据加工程度`、`影响对象`、`影响程度`、`数据级别` 理解上下文，但 `proposed_description` 不要写推荐分级，不要写“定级为几级”，不要写“依据文档定级为……”。
- 不要把分类说明写成分级理由；不要写影响程度，不要写危害后果，不要写“泄露后可能……”“造成严重危害”“特别严重危害”等风险结论。
- 只有当分类路径和数据范围都不足以判断分类对象、业务场景或数据内容时，才返回 `证据不足，无法从当前文档确定`。
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
