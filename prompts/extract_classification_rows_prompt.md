# 任务

从输入 `input_payload.table_segments` 中抽取候选分类分级明细行 `classification_rows`。只返回 JSON object。

# 必须遵守

- 只从每个 segment.text 的文本证据中抽取，不得使用文档外知识。
- 不要生成分类树；只抽取文档中已经出现的候选明细行。
- 不要硬编码行业分类、业务类别、安全等级或示例。
- `path_levels` 必须来自文档原文中的分类路径、表格列、标题层级或相邻结构。
- `recommended_grade` 只有在文档明确给出时才能填写；否则返回 `null`。
- `description` 优先使用原文说明；如果只有零散证据，可以做证据内总结；如果证据不足，必须返回 `证据不足，无法从当前文档确定`。
- `description_source` 只能是 `quoted`、`summarized`、`insufficient`。
- `evidence_quote` 必须是支持该 row 的 segment.text 原文片段。
- `evidence_chunk_ids` 必须填写证据来源的 `source_chunk_id`，不是 `segment_id`。
- 如果 `evidence_quote` 不能完整覆盖 `description`，用 `description_evidence_quote` 单独给出说明证据。
- 结构推断、弱证据、无明确分级、说明不足都必须设置 `needs_review=true`。
- 输出字段名和枚举值保持英文，不要翻译 JSON key。
- 必须抽取本批次所有可识别的分类分级明细行；不要只抽取示例、代表行或摘要行。
- 如果一个 segment 是续表，分类路径中的上级 `类`、`项`、`目` 可以从本 segment 的相邻结构、上一行或表格上下文继承；继承属于 structural support，必须保留证据并在必要时 `needs_review=true`。
- 每个 segment 可能包含 `structure_context`。这是版式上下文，不是业务答案；只能用于理解附录、分类标题、表题、续表、层级表头、页码和行号。
- `structure_context.table_title`、`structure_context.classification_title`、`structure_context.appendix_heading` 可以帮助恢复续表或拆分 segment 缺失的上级路径；如果仅依赖这些字段，`support_level` 应为 `structural` 且通常需要 `needs_review=true`。
- `structure_context.hierarchy_header` 和 `header_text` 只表示表格列头，不得把 `类`、`项`、`目`、`数据范围及示例`、`数据级别` 等列名当作 `path_levels`。
- 每个 segment 可能包含 `flattened_row_hints`。这是版式提示，不是业务答案；只能用于理解同一行里是否有多个层级编号。
- 如果 `flattened_row_hints` 显示同一原文行中有多个层级编号，例如 `2.5 ... 2.5.7 ...`，不要把整行直接放进一个 `path_levels` 元素；应优先按原文编号结构拆成多级路径。
- 如果依赖 `flattened_row_hints` 拆分路径，`support_level` 应为 `structural`，并设置 `needs_review=true`。
- 当表格中出现 `数据范围及示例`、`数据加工程度`、`影响对象`、`影响程度`、`数据级别` 等列时，分别填入 `data_range_examples`、`processing_degree`、`impact_object`、`impact_degree`、`recommended_grade`。
- `grade_evidence_quote` 必须覆盖推荐分级及其相邻分级因素，例如“原始数据 个人 严重危害 一般数据3级”。
- 同一分类路径下如果出现多条数据范围或多个分级，不要丢弃；可以输出多条候选行，后续 normalization 会合并。
- 如果本批次文本中只有下级项目，缺少上级路径，必须尽量从 segment 的 `structure_context`、`header_text`、`section_title` 或相邻行恢复；无法恢复时输出可证据支持的部分路径并 `needs_review=true`。

# 输出格式

返回如下 JSON object：

```json
{
  "classification_rows": [
    {
      "path_levels": [],
      "recommended_grade": null,
      "data_range_examples": [],
      "processing_degree": "",
      "impact_object": "",
      "impact_degree": "",
      "description": "",
      "description_source": "quoted",
      "description_evidence_quote": "",
      "grade_evidence_quote": "",
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
