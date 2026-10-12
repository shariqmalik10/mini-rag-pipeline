import shutil
import unittest

from rag.ingest import STRATEGIES, chunk_documents, load_documents, parse_document, sentence_spans
from tests.helpers import SAMPLE_KB, TempDirTest


class ParseDocumentTest(unittest.TestCase):
    def test_header_variants_parse_the_same(self):
        clean = parse_document("Title: Fees\nSection: Payments\n\nCards are free.\n", "a")
        messy = parse_document("﻿section:Payments\r\nTITLE:   Fees \r\n\r\nCards are free.\r\n", "b")
        self.assertEqual((clean.title, clean.section, clean.body), (messy.title, messy.section, messy.body))

    def test_missing_section_is_empty(self):
        self.assertEqual(parse_document("Title: Fees\n\nCards are free.", "a").section, "")

    def test_missing_or_empty_title_fails_naming_the_file(self):
        for raw in ("Section: Payments\n\nCards are free.", "Title:\nSection: Payments\n\nCards are free."):
            with self.assertRaisesRegex(ValueError, "fees.txt: missing or empty 'Title:'"):
                parse_document(raw, "fees.txt")

    def test_title_characters_that_break_citations_are_replaced(self):
        self.assertEqual(parse_document("Title: Fees [draft] § 2\n\nBody.", "a").title, "Fees (draft) S 2")

    def test_extra_header_lines_stay_out_of_the_body(self):
        doc = parse_document("Title: Leave\nCategory: HR\nSection: Policies\n\nStaff get 25 days.", "a")
        self.assertEqual((doc.section, doc.body), ("Policies", "Staff get 25 days."))

    def test_header_like_line_in_body_stays_in_body(self):
        doc = parse_document("Title: A\n\nNote: see below.\nTitle: not a header.", "a")
        self.assertIn("Title: not a header.", doc.body)


class SentenceSplitTest(unittest.TestCase):
    def split(self, body):
        return [body[s:e] for s, e in sentence_spans(body)]

    def test_does_not_split_inside_times_abbreviations_or_decimals(self):
        body = "Requests before 17:00 UTC are reviewed, e.g. same day. Fees are 1.5 EUR. Done."
        self.assertEqual(self.split(body), [
            "Requests before 17:00 UTC are reviewed, e.g. same day.", "Fees are 1.5 EUR.", "Done."])

    def test_pronoun_led_sentence_joins_the_previous_one(self):
        self.assertEqual(self.split("Payouts take 4 hours. This applies to verified accounts.\nOther fact."),
                         ["Payouts take 4 hours. This applies to verified accounts.", "Other fact."])

    def test_pronoun_led_sentence_after_blank_line_stands_alone(self):
        self.assertEqual(self.split("Payouts take 4 hours.\n\nThis is a new paragraph."),
                         ["Payouts take 4 hours.", "This is a new paragraph."])

    def test_single_letters_and_numbered_items(self):
        self.assertEqual(self.split("Take vitamin C. Do not exceed 2 tablets. See the U.S. office.\n1. Return within 30 days.\n2. Keep the receipt."),
                         ["Take vitamin C.", "Do not exceed 2 tablets.", "See the U.S. office.",
                          "1. Return within 30 days.", "2. Keep the receipt."])

    def test_long_unpunctuated_text_is_split_into_bounded_chunks(self):
        body = " ".join(["word"] * 400)
        spans = sentence_spans(body)
        self.assertGreater(len(spans), 1)
        self.assertTrue(all(e - s <= 500 for s, e in spans))
        self.assertEqual(" ".join(body[s:e] for s, e in spans), body)

    def test_bullets_and_wrapped_lines(self):
        self.assertEqual(self.split("Rules\n- one item\n- two item\nA wrapped\nsentence."),
                         ["Rules", "- one item", "- two item\nA wrapped\nsentence."])


class ChunkingTest(TempDirTest):
    def test_offsets_reproduce_chunk_text_for_every_strategy(self):
        docs = {d.title: d for d in load_documents(SAMPLE_KB)}
        for strategy in STRATEGIES:
            for chunk in chunk_documents(list(docs.values()), strategy):
                body = docs[chunk.doc_title].body
                self.assertEqual(body[chunk.start_char:chunk.end_char], chunk.text, (strategy, chunk.chunk_id))
                self.assertTrue(chunk.text.strip())

    def test_renamed_nested_and_shuffled_files_give_identical_chunks(self):
        sources = sorted(SAMPLE_KB.iterdir())
        for i, src in enumerate(reversed(sources)):
            dest = self.tmp / f"group{i % 2}" / f"zz_{i}.md"
            dest.parent.mkdir(exist_ok=True)
            shutil.copy(src, dest)
        (self.tmp / ".DS_Store").write_bytes(b"\x00\x01binary")
        (self.tmp / ".hidden").mkdir()
        (self.tmp / ".hidden" / "notes.txt").write_text("no title here")
        for strategy in STRATEGIES:
            original = [c.to_dict() for c in chunk_documents(load_documents(SAMPLE_KB), strategy)]
            moved = [c.to_dict() for c in chunk_documents(load_documents(self.tmp), strategy)]
            self.assertEqual(original, moved, strategy)

    def test_duplicate_titles_keep_chunk_ids_unique(self):
        (self.tmp / "a.txt").write_text("Title: Fees\n\nOne. Two.")
        (self.tmp / "b.txt").write_text("Title: Fees\n\nThree.")
        ids = [c.chunk_id for c in chunk_documents(load_documents(self.tmp), "sentence")]
        self.assertEqual(sorted(ids), ["chunk_1", "chunk_2", "chunk_3"])

    def test_empty_kb_fails(self):
        with self.assertRaisesRegex(ValueError, "no usable documents"):
            load_documents(self.tmp)

    def test_unusable_files_are_skipped_and_other_encodings_read(self):
        shutil.copytree(SAMPLE_KB, self.tmp, dirs_exist_ok=True)
        (self.tmp / "README").write_text("Notes for maintainers, no title.")
        (self.tmp / "logo.bin").write_bytes(b"\x89PNG\x00\x00binary")
        (self.tmp / "latin.txt").write_bytes("Title: Caf\xe9 hours\n\nThe caf\xe9 opens at 8.".encode("cp1252"))
        (self.tmp / "wide.txt").write_bytes("Title: Parking\n\nParking is free.".encode("utf-16"))
        with self.assertLogs("rag", level="WARNING") as logs:
            titles = {d.title for d in load_documents(self.tmp)}
        self.assertEqual(titles - {d.title for d in load_documents(SAMPLE_KB)}, {"Café hours", "Parking"})
        self.assertTrue(any("README" in line for line in logs.output))
        self.assertTrue(any("logo.bin: binary file" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
