# Bedrock Managed Knowledge Base Support

## Overview
Adds an OpenHands knowledge provider that queries Amazon Bedrock Knowledge Bases for managed retrieval during agent tasks.

## Usage
```python
from openhands.knowledge.bedrock_kb import BedrockKnowledgeBaseProvider

provider = BedrockKnowledgeBaseProvider(
    knowledge_base_id="YOUR_KB_ID",
    region="us-east-1",
)
results = provider.search("How do I fix a failing deployment?")
for doc in results:
    print(doc.content, doc.score)
```

## Configuration
| Variable | Description | Default |
|---|---|---|
| KNOWLEDGE_BASE_ID | Bedrock Knowledge Base ID | None |
| AWS_REGION | AWS region for the KB | us-east-1 |
| AWS_PROFILE | AWS credentials profile | None |
| USE_AGENTIC_RETRIEVAL | Enable agentic retrieval | true |
| MAX_RESULTS | Maximum retrieval results | 5 |

## Features
- Managed search (no vector store needed)
- Agentic retrieval with query decomposition + reranking
- Automatic fallback to plain Retrieve if agentic fails
- Multi-source support (S3, Web, Confluence, SharePoint)
- Integrates with OpenHands agent runtime context

## SDK Requirements
- boto3 >= 1.43
- openhands >= 0.1

## Required IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieveStream"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```

## References
- [Build a Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html)
- [Retrieve API](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html)
- [Agentic Retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-agentic.html)
