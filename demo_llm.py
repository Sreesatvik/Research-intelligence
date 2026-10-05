from generation.groq_llm import GroqLLM

llm = GroqLLM()
print(llm.generate("You answer in one short sentence.", "What is an embedding?"))
