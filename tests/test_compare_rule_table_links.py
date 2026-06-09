from __future__ import annotations

import unittest

from scripts.compare_rule_table_links import parse_reference_spec


class CompareRuleTableLinksTests(unittest.TestCase):
    def test_parse_reference_spec_accepts_labeled_reference(self) -> None:
        reference = parse_reference_spec("outputs_233/rule_table.json:existing_rule_table:233国标")

        self.assertEqual(reference.path, "outputs_233/rule_table.json")
        self.assertEqual(reference.source_type, "existing_rule_table")
        self.assertEqual(reference.name, "233国标")
        self.assertEqual(reference.rows, [])

    def test_parse_reference_spec_rejects_missing_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "PATH:TYPE:NAME"):
            parse_reference_spec("outputs_233/rule_table.json")


if __name__ == "__main__":
    unittest.main()
