# 任务

对候选 classification_rows 做证据内归一。

注意：当前代码实现是确定性的 `normalize_classification_rows`，不会调用 LLM。本 prompt 仅供未来需要 LLM 辅助精修时使用。

# 允许操作

- 合并重复的 classification_rows。
- 去除 path_levels 中的空白层级，并规范首尾空白。
- 标记证据不足的行。
- 在已有候选之间选择证据更强的一行。

# 严格约束

- 不要新增 path_levels。
- 不要新增 recommended_grade。
- 不要编造文档外分类说明。
- 不要根据常识补充分级、分类路径或描述。
- 证据不足的说明必须精确使用：`证据不足，无法从当前文档确定`。
- 如果输入证据不足，必须保留 needs_review=true，并给出非空 review_reason。

# 输出要求

只返回 JSON object，不要 Markdown，不要解释文字。
