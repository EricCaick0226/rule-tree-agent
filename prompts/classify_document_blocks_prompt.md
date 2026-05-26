你要为输入的每个 document chunk 标注一个非破坏性的 `block_signal`，帮助后续步骤判断文本块更像哪类证据。

允许的 `block_signal` 只能是：
- `table_like`：像分类表、字段表或带多列结构的行列内容。
- `hierarchy_like`：像分类目录、层级清单、编号层级或父子关系描述。
- `grade_legend`：像等级定义、等级说明、共享/开放/敏感程度说明。
- `prose_rule`：像规则条款、适用条件、纳入/排除范围或文字化判断依据。
- `normal`：普通说明文本，当前没有明显特殊结构。
- `possible_noise`：页眉页脚、乱码、重复碎片、格式噪声或证据价值很弱的内容。

硬性约束：
- 不得删除、合并、拆分或忽略 chunk；每个输入 chunk 都应返回一个信号。
- `block_signal` 只是供后续步骤参考的信号，不是最终分类事实、等级事实或规则事实。
- 不要生成分类树，不要抽取最终分类行，不要推断等级映射。
- 只能使用当前输入 chunk 的文本和元数据，不得使用外部知识、行业常识、默认分类或默认等级。
- 证据弱、文本含混、OCR 可疑、结构不完整或你不确定时，必须设置 `needs_review=true` 并说明 `review_reason`。
- 输出必须是一个 JSON object，不要 Markdown，不要解释文字。

返回格式：
```json
{
  "block_signals": [
    {
      "chunk_id": "输入中的 chunk_id",
      "block_signal": "table_like",
      "reason": "基于当前 chunk 文本的简短理由",
      "confidence": 0.0,
      "needs_review": false,
      "review_reason": ""
    }
  ]
}
```
