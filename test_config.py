import ast
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BOT_PATH = PROJECT_DIR / "bot.py"


class ConfigurationTests(unittest.TestCase):
    def test_bot_uses_settings_module_and_never_reads_token_from_backup_source(self):
        source = BOT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "settings"
            for alias in node.names
        }

        self.assertNotIn("bot_backup.py", source)
        self.assertIn("TELEGRAM_TOKEN", imported_names)
        self.assertIn("MARKS_DB_PATH", imported_names)
        self.assertIn("GEMINI_CHAT_MODEL", imported_names)


if __name__ == "__main__":
    unittest.main()
