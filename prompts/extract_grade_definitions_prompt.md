# 任务

从输入 chunk 中抽取文档明确给出的分级定义、等级名称、共享属性、开放属性或分级标准。只返回 JSON。

# 约束

- 只能抽取 chunk 原文中明确出现的内容，不得使用文档外知识、行业常识或默认规则。
- 只抽取文档中出现的等级名称，不允许预设等级体系，也不得补全未出现的等级。
- 只抽取分级定义、等级名称、共享属性、开放属性或分级标准；不要把分类路径、目录层级、表头路径当作等级定义。
- 如果只看到等级名称但没有定义，definition 可以复用原文短句，并且必须标记 needs_review=true。
- evidence_quote 必须逐字来自输入 chunk，不能改写、概括或拼接文档外文本。
- evidence_chunk_ids 只能填写输入中存在的 chunk_id。
- 证据不足时输出空数组，或将对应项标记 needs_review=true、status="proposed"。

# 输出

{
  "grade_definitions": [
    {
      "grade_name": "...",
      "definition": "...",
      "criteria": ["..."],
      "evidence_quote": "...",
      "evidence_chunk_ids": ["doc_1_chunk_1"],
      "confidence": 0.0,
      "needs_review": true,
      "review_reason": "...",
      "status": "evidence_supported | proposed"
    }
  ]
}
