import unittest

from vla_grounder.protocol import extract_evaluation_command, extract_training_command


class CommandParserTest(unittest.TestCase):
    def test_answer_has_priority(self):
        text = "<think>reasoning</think><answer>pick the cup</answer>ignored"
        self.assertEqual(extract_training_command(text), "pick the cup")
        self.assertEqual(extract_evaluation_command(text), "pick the cup")

    def test_content_after_thinking_is_used(self):
        text = "<think>reasoning</think> place the carrot on the plate"
        self.assertEqual(extract_training_command(text), "place the carrot on the plate")
        self.assertEqual(extract_evaluation_command(text), "place the carrot on the plate")

    def test_training_fallback_stops(self):
        self.assertEqual(extract_training_command("unfinished reasoning"), "stop")

    def test_evaluation_fallback_keeps_last_100_characters(self):
        text = "x" * 120
        self.assertEqual(extract_evaluation_command(text), "x" * 100)

    def test_parser_length_limits_match_original_experiments(self):
        answer = "x" * 220
        text = f"<answer>{answer}</answer>"
        self.assertEqual(extract_training_command(text), "x" * 200)
        self.assertEqual(extract_evaluation_command(text), "x" * 150)


if __name__ == "__main__":
    unittest.main()
