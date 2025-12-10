# AI Badgr Provider

AI Badgr is a cheaper OpenAI-compatible backend that can be used as a drop-in replacement for OpenAI's API. This guide shows you how to configure OpenHands to use AI Badgr.

## Overview

AI Badgr implements the OpenAI API specification, making it compatible with existing OpenAI client libraries. You can use it by simply overriding the `base_url` parameter in your configuration.

**Key Benefits:**
- Significantly lower costs compared to OpenAI
- Full OpenAI API compatibility
- Supports streaming and JSON mode
- Works with standard OpenAI SDKs

## Configuration

### Environment Variables

Set the following environment variables:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_BASE_URL=https://aibadgr.com/api/v1
```

### Python Usage

Using the `openai` package:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://aibadgr.com/api/v1"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200
)
print(response.choices[0].message.content)
```

### JavaScript/Node.js Usage

Using the `openai` package:

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'YOUR_API_KEY',
  baseURL: 'https://aibadgr.com/api/v1',
});

const response = await client.chat.completions.create({
  model: 'gpt-3.5-turbo',
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 200,
});

console.log(response.choices[0].message.content);
```

### cURL Usage

Command-line usage:

```bash
curl https://aibadgr.com/api/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```

## Advanced Features

### Streaming

Enable streaming responses:

```python
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

Request structured JSON responses:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Return a JSON object"}],
    response_format={"type": "json_object"}
)
```

## Integration with OpenHands

To use AI Badgr with OpenHands, configure your `config.toml` or set the appropriate environment variables. OpenHands uses LiteLLM, which supports OpenAI-compatible endpoints through the `base_url` parameter.

For more information on configuring LLM providers in OpenHands, see the [LLM configuration documentation](../openhands/llm/README.md).

