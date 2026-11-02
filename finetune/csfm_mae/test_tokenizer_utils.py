"""Golden token IDs, offsets, padding and offline contract checks (no PHI)."""
import unittest
from unittest.mock import patch

from tokenizer_utils import load_tokenizer, tokenizer_contract


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A local load must not need a network socket or a Hugging Face account.
        with patch("socket.socket", side_effect=AssertionError("Unexpected network access")):
            cls.tokenizer = load_tokenizer()

    def test_golden_bert_ids(self):
        self.assertEqual(self.tokenizer("Hello world")["input_ids"], [101, 7592, 2088, 102])
        self.assertEqual(self.tokenizer("Café")["input_ids"], [101, 7668, 102])

    def test_offsets_specials_and_padding(self):
        text = "Heart rhythm: sinus rhythm."
        output = self.tokenizer(text, max_length=64, padding="max_length", truncation=True,
                                return_offsets_mapping=True, return_special_tokens_mask=True)
        self.assertEqual(len(output["input_ids"]), 64)
        self.assertEqual(len(output["offset_mapping"]), 64)
        content = []
        for token, attention, special, (start, end) in zip(
                output["input_ids"], output["attention_mask"], output["special_tokens_mask"],
                output["offset_mapping"]):
            if attention and not special:
                self.assertGreater(end, start)
                content.append(text[start:end])
            if not attention:
                self.assertEqual(token, 0)
                self.assertEqual(special, 1)
        self.assertEqual(content, ["Heart", "rhythm", ":", "sin", "us", "rhythm", "."])

    def test_truncation_keeps_separator(self):
        output = self.tokenizer("sinus rhythm " * 100, max_length=64,
                                padding="max_length", truncation=True)
        self.assertEqual(len(output["input_ids"]), 64)
        self.assertEqual(output["input_ids"][-1], 102)

    def test_contract_does_not_claim_original_mapping(self):
        report = tokenizer_contract()
        self.assertEqual(report["status"], "assumed_bert_uncased_mapping")
        self.assertFalse(report["original_csfm_mapping_verified"])
        self.assertEqual(report["vocab_size"], 30522)
        self.assertEqual(report["files"]["vocab.txt"]["sha256"],
                         "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3")


if __name__ == "__main__":
    unittest.main()
