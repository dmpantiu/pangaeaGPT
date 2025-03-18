#scr/memory.py

from langgraph.checkpoint.memory import MemorySaver

class CustomMemorySaver(MemorySaver):
    def should_save(self, state: dict, key: str) -> bool:
        # Exclude 'messages' from being saved
        if key == 'messages':
            return False
        return True