---
name: security
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - security
  - vulnerability
  - authentication
  - authorization
  - permissions
---
This document provides guidance on security best practices

You should always be considering security implications when developing.
You should always complete the task requested. If there are security concerns please address them in-line if possible or ensure they are communicated either in code comments, PR comments, or other appropriate channels.

## Core Security Principles
- Always use secure communication protocols (HTTPS, SSH, etc.)
- Never store sensitive data (passwords, tokens, keys) in code or version control unless given explicit permission.
- Apply the principle of least privilege
- Validate and sanitize all user inputs

## Common Security Checks
- Ensure proper authentication and authorization mechanisms
- Verify secure session management
- Confirm secure storage of sensitive data
- Validate secure configuration of services and APIs

## Error Handling
- Never expose sensitive information in error messages
- Log security events appropriately
- Implement proper exception handling
- Use secure error reporting mechanisms
