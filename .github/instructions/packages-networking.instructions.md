---
applyTo: "packages/hmls-server/**,packages/hmls-client/**,packages/hmls-observer/**,packages/hmls-networking/**,packages/hmls-protocol/**"
---

# Networking Packages

## Package Roles

| Package | Purpose |
|---------|---------|
| `hmls-protocol` | Pydantic message models for client↔server communication |
| `hmls-networking` | Shared WebSocket utilities (connection handling, message framing) |
| `hmls-server` | Game server — hosts a match, manages players and observers |
| `hmls-client` | Human player client with Textual TUI |
| `hmls-observer` | Spectator client (sees full map, no fog-of-war) |

## Architecture

- Communication is over WebSockets (using the `websockets` library).
- All messages are Pydantic models serialised as JSON.
- The server runs one game at a time; it waits for two players to connect.
- Observers can connect at any time.

## Protocol

Message types are defined in `hmls-protocol`. When adding new message types:
1. Define the Pydantic model in `hmls.protocol.messages`.
2. Ensure both client and server handle the new message.
3. Protocol changes must be backward-compatible where possible.
