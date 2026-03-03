# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
class PatchingException(Exception):
    pass


class HunkException(PatchingException):
    def __init__(self, msg: str, hunk: int | None = None) -> None:
        self.hunk = hunk
        if hunk is not None:
            super().__init__('{msg}, in hunk #{n}'.format(msg=msg, n=hunk))
        else:
            super().__init__(msg)


class ApplyException(PatchingException):
    pass


class SubprocessException(ApplyException):
    def __init__(self, msg: str, code: int) -> None:
        super().__init__(msg)
        self.code = code


class HunkApplyException(HunkException, ApplyException, ValueError):
    pass


class ParseException(HunkException, ValueError):
    pass
