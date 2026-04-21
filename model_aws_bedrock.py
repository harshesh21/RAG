import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt = "Explain what a context window is in one paragraph, simply."

response = client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    })
)

result = json.loads(response["body"].read())

# The actual response
print("Response:")
print(result["content"][0]["text"])

# Token usage
print("\nToken usage:")
print(f"  Input tokens:  {result['usage']['input_tokens']}")
print(f"  Output tokens: {result['usage']['output_tokens']}")
print(f"  Total tokens:  {result['usage']['input_tokens'] + result['usage']['output_tokens']}")