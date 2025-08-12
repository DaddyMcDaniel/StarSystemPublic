#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


MEM_DIR = Path(__file__).resolve().parent
MEM_FILE = MEM_DIR / "agent_b_story_memory.json"


STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","at","is","it","this","that","for","with","as","by","be","are","was","were","from","you","your","we","our","i","he","she","they","them","his","her","their","not","no","yes","press","enter","continue","start","menu","game","play","level","world","map","option","settings","editor","pause","resume","back","select","up","down","left","right"
}


def tokenize(text: str) -> List[str]:
    t = text.lower()
    # keep letters and digits
    words = re.findall(r"[a-z0-9]{3,}", t)
    return [w for w in words if w not in STOPWORDS and not w.isdigit()]


@dataclass
class StorySnapshot:
    text: str
    keywords: List[str]
    context: str


class StoryMemory:
    def __init__(self):
        self.story_lines: List[StorySnapshot] = []
        self.keywords_counter: Counter = Counter()
        self.ui_context_stats: Dict[str, int] = {}
        self._load()

    def _load(self):
        try:
            if MEM_FILE.exists():
                obj = json.loads(MEM_FILE.read_text())
                self.story_lines = [StorySnapshot(**s) for s in obj.get("story_lines", [])]
                self.keywords_counter = Counter(obj.get("keywords_counter", {}))
                self.ui_context_stats = dict(obj.get("ui_context_stats", {}))
        except Exception:
            pass

    def _save(self):
        try:
            MEM_DIR.mkdir(parents=True, exist_ok=True)
            obj = {
                "story_lines": [s.__dict__ for s in self.story_lines][-500:],
                "keywords_counter": dict(self.keywords_counter),
                "ui_context_stats": self.ui_context_stats,
            }
            MEM_FILE.write_text(json.dumps(obj, indent=2))
        except Exception:
            pass

    def add_text(self, text: str, context: str = "unknown") -> List[str]:
        kws = tokenize(text)
        if kws:
            self.keywords_counter.update(kws)
        self.ui_context_stats[context] = int(self.ui_context_stats.get(context, 0)) + 1
        self.story_lines.append(StorySnapshot(text=text, keywords=kws, context=context))
        self._save()
        return kws

    def summary(self, top_k: int = 15) -> Dict[str, any]:
        top_keywords = [w for w, _ in self.keywords_counter.most_common(top_k)]
        return {
            "top_keywords": top_keywords,
            "ui_contexts": dict(self.ui_context_stats),
            "lines": [s.text for s in self.story_lines[-20:]],
        }


