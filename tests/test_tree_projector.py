from __future__ import annotations

import unittest

from src.core.agent_state import AgentState, ClassificationRow, EvidenceRef
from src.steps.tree_projector import project_tree_from_rows


def _ref(evidence_id: str, chunk_id: str | None = None) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        chunk_id=chunk_id or evidence_id,
        doc_name="policy.txt",
        section_title="分类表",
        text=f"{evidence_id} text",
        used_for="classification_row",
        relevance_score=0.9,
    )


class TreeProjectorTests(unittest.TestCase):
    def test_projects_tree_without_llm_generated_nodes(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade="3级",
                    description="患者说明",
                    description_source="quoted",
                    evidence_quote="基础资源 服务范围与对象 患者 3级 患者说明",
                    support_level="explicit",
                    needs_review=False,
                )
            ],
        )

        result = project_tree_from_rows(state)

        paths = {node.path: node for node in result.nodes}
        self.assertIn("基础资源", paths)
        self.assertIn("基础资源 / 服务范围与对象", paths)
        self.assertIn("基础资源 / 服务范围与对象 / 患者", paths)
        leaf = paths["基础资源 / 服务范围与对象 / 患者"]
        self.assertEqual(leaf.grade, "3级")
        self.assertEqual(leaf.description, "患者说明")

    def test_rows_sharing_prefix_reuse_parent_and_sort_paths(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础资源", "服务范围与对象", "医务人员"],
                    recommended_grade="2级",
                    description="医务人员说明",
                    support_level="explicit",
                    needs_review=False,
                ),
                ClassificationRow(
                    row_id="row_2",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade="3级",
                    description="患者说明",
                    support_level="explicit",
                    needs_review=False,
                ),
            ],
        )

        result = project_tree_from_rows(state)

        paths = [node.path for node in result.nodes]
        self.assertEqual(paths, sorted(paths))
        self.assertEqual(len(paths), len(set(paths)))

        path_map = {node.path: node for node in result.nodes}
        parent = path_map["基础资源 / 服务范围与对象"]
        self.assertEqual(
            path_map["基础资源 / 服务范围与对象 / 医务人员"].parent_id,
            parent.node_id,
        )
        self.assertEqual(
            path_map["基础资源 / 服务范围与对象 / 患者"].parent_id,
            parent.node_id,
        )

    def test_missing_grade_marks_leaf_for_review(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade=None,
                    description="患者说明",
                    support_level="explicit",
                    needs_review=False,
                )
            ],
        )

        result = project_tree_from_rows(state)

        leaf = {
            node.path: node for node in result.nodes
        }["基础资源 / 服务范围与对象 / 患者"]
        self.assertIsNone(leaf.grade)
        self.assertEqual(leaf.grade_reason, "证据不足，无法从当前文档确定推荐分级。")
        self.assertTrue(leaf.needs_review)

    def test_shared_parent_merges_evidence_and_review_conservatively(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade="3级",
                    description="患者说明",
                    evidence_refs=[_ref("ev_1")],
                    confidence=0.9,
                    needs_review=False,
                    status="evidence_supported",
                ),
                ClassificationRow(
                    row_id="row_2",
                    path_levels=["基础资源", "服务范围与对象", "医务人员"],
                    recommended_grade="2级",
                    description="医务人员说明",
                    evidence_refs=[_ref("ev_2")],
                    confidence=0.4,
                    needs_review=True,
                    status="proposed",
                ),
            ],
        )

        result = project_tree_from_rows(state)

        parent = {
            node.path: node for node in result.nodes
        }["基础资源 / 服务范围与对象"]
        self.assertTrue(parent.needs_review)
        self.assertEqual({ref.evidence_id for ref in parent.evidence_refs}, {"ev_1", "ev_2"})
        self.assertEqual(parent.confidence, 0.4)
        self.assertEqual(parent.status, "proposed")

    def test_duplicate_leaf_path_merges_evidence_and_review_conservatively(self) -> None:
        state = AgentState(
            task="test",
            classification_rows=[
                ClassificationRow(
                    row_id="row_1",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade="3级",
                    description="患者说明一",
                    evidence_refs=[_ref("ev_1"), _ref("same_evidence", "chunk_a")],
                    confidence=0.8,
                    needs_review=False,
                    status="evidence_supported",
                ),
                ClassificationRow(
                    row_id="row_2",
                    path_levels=["基础资源", "服务范围与对象", "患者"],
                    recommended_grade="3级",
                    description="患者说明二",
                    evidence_refs=[
                        _ref("ev_2"),
                        _ref("same_evidence", "chunk_b"),
                        _ref("ev_3", "chunk_a"),
                    ],
                    confidence=0.3,
                    needs_review=True,
                    status="proposed",
                ),
            ],
        )

        result = project_tree_from_rows(state)

        leaf = {
            node.path: node for node in result.nodes
        }["基础资源 / 服务范围与对象 / 患者"]
        self.assertEqual(leaf.description, "患者说明二")
        self.assertEqual({ref.evidence_id for ref in leaf.evidence_refs}, {"ev_1", "same_evidence", "ev_2"})
        self.assertEqual(
            {ref.evidence_id for ref in leaf.description_evidence_refs},
            {"ev_1", "same_evidence", "ev_2"},
        )
        self.assertEqual(
            {ref.evidence_id for ref in leaf.grade_evidence_refs},
            {"ev_1", "same_evidence", "ev_2"},
        )
        self.assertTrue(leaf.needs_review)
        self.assertEqual(leaf.confidence, 0.3)
        self.assertEqual(leaf.status, "proposed")


if __name__ == "__main__":
    unittest.main()
