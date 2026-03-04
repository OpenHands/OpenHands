"""This class is similar to the RuntimeStatus defined in the runtime api. (When this class was defined
a RuntimeStatus class already existed in OpenHands which serves a completely different purpose) Some of
the status definitions do not match up:

STOPPED/paused - the runtime is not running but may be restarted
ARCHIVED/stopped - the runtime is not running and will not restart due to deleted files.
"""

from enum import Enum


class ConversationStatus(Enum):
    # The conversation is starting
    STARTING = 'STARTING'
    # The conversation is running - the agent may be working or idle
    RUNNING = 'RUNNING'
    # The conversation has stopped (This is synonymous with `paused` in the runtime API.)
    STOPPED = 'STOPPED'
    # The conversation has been archived and cannot be restarted.
    ARCHIVED = 'ARCHIVED'
    # Something has gone wrong with the conversation (The runtime rather than the agent)
    ERROR = 'ERROR'
