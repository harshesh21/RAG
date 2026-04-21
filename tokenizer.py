import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # same encoding Claude/GPT-4 use

texts = [
    "The cat sat on the mat.",
    "tokenization",
    "SELECT * FROM users WHERE id = 12345",
    "def retrieve_documents(query, top_k=5):",
]

print("Token counts:\n")
for text in texts:
    tokens = enc.encode(text)
    print(f"  Text:   {text}")
    print(f"  Tokens: {tokens}")
    print(f"  Count:  {len(tokens)}\n")