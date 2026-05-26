# 任务

从输入 chunk 中抽取候选分类分级明细行 `classification_rows`。只返回 JSON object。

# 必须遵守

- 只从输入 chunk 的文本证据中抽取，不得使用文档外知识。
- 不要生成分类树；只抽取文档中已经出现的候选明细行。
- 不要硬编码行业分类、业务类别、安全等级或示例。
- `path_levels` 必须来自文档原文中的分类路径、表格列、标题层级或相邻结构。
- `recommended_grade` 只有在文档明确给出时才能填写；否则返回 `null`。
- `description` 优先使用原文说明；如果只有零散证据，可以做证据内总结；如果证据不足，必须返回 `证据不足，无法从当前文档确定`。
- `description_source` 只能是 `quoted`、`summarized`、`insufficient`。
- `evidence_quote` 必须是支持该 row 的输入 chunk 原文片段。
- 如果 `evidence_quote` 不能完整覆盖 `description`，用 `description_evidence_quote` 单独给出说明证据。
- 结构推断、弱证据、无明确分级、说明不足都必须设置 `needs_review=true`。
- 输出字段名和枚举值保持英文，不要翻译 JSON key。

# 输出格式

返回如下 JSON object：

```json
{
  "classification_rows": [
    {
      "path_levels": [],
      "recommended_grade": null,
      "description": "",
      "description_source": "quoted",
      "description_evidence_quote": "",
      "evidence_quote": "",
      "evidence_chunk_ids": [],
      "support_level": "explicit",
      "confidence": 0.0,
      "needs_review": true,
      "review_reason": "",
      "status": "evidence_supported"
    }
  ]
}
```
