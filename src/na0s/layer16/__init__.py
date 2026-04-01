"""Layer 16 — Multi-Turn Detection.

Adds conversation-level memory and stateful analysis to Na0S. Detects
multi-turn attacks where adversaries spread payloads across messages,
gradually escalate, plant context in early turns, or fabricate history.

ARCHITECTURE DECISION: Post-Processor Pattern (Option C).
Layer 16 runs AFTER single-turn scan() completes. When a session_id
is provided, it records the turn, runs multi-turn detectors on the
accumulated conversation state, and merges alerts into ScanResult.
The existing stateless API is unchanged — session_id is optional.

Leverages existing Na0S infrastructure:
- multi_turn_validator.py (rolling window, escalation streak)
- payload_assembly_detector.py (D7.2 fragment assembly)
- ScanResult (extensible dataclass)

Components
----------
- **ConversationSecurityMonitor** (conversation_monitor.py): Main entry point
- **ConversationState** (state.py): Per-session conversation memory
- **SessionManager** (session_manager.py): Session lifecycle management
- **SlidingWindow** (sliding_window.py): Bounded turn history
- **Detectors** (detectors/): Escalation, payload splitting, fabricated history
- **Storage** (storage/): Memory, SQLite, Redis backends
- **Testing** (testing/): Multi-turn test harness and metrics
"""
