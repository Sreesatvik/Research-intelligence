from agent.setup import make_agent

agent = make_agent()
q = ("Find papers about defending against prompt injection, and tell me what research topic "
     "the classifier assigns to the best match.")
result = agent.run(q)
print("Q:", q)
for c in result.calls:
    print(f"  step {c.step}: {c.tool} {c.arguments} -> {c.result_chars} chars, {'ERROR' if c.is_error else 'ok'}, {c.seconds}s")
print("stopped because:", result.stopped_because)
print("\nANSWER:\n", result.answer)
