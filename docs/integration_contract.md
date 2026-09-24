# VeriMedia AI Integration Contract

## Purpose

Every forensic module must return a standardized `ModuleResult`.

This contract is used by:

- Image forensics
- Video forensics
- Audio forensics
- Lip-sync analysis
- Multimodal fusion

The backend uses this structure to validate, combine, store, and explain forensic results.

## Supported media types

The `media_type` field must use one of these values:

```text
image
video
audio
lipsync
multimodal