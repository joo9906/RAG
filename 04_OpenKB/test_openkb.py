# pyrefly: ignore [missing-import]
from openkb import KnowledgeBase

kb = KnowledgeBase("./.openkb")

answer = kb.aquery("라이트라그 구현하려면 뭐부터 해야해?")
print(answer.output)

retrieve = kb.retrieve("라이트라그 구현하려면 뭐부터 해야해?")
print(retrieve.retrieved_docs)
