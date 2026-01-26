from abc import ABC, abstractmethod

from integrations.models import Message, SourceType


class Manager(ABC):
    manager_type: SourceType

    @abstractmethod
    async def receive_message(self, message: Message):
        "Receive message from integration"
        raise NotImplementedError

    @abstractmethod
    def send_message(self, message: Message):
        "Send message to integration from Openhands server"
        raise NotImplementedError

    @abstractmethod
    def start_job(self):
        "Kick off a job with openhands agent"
        raise NotImplementedError

    def create_outgoing_message(self, msg: str | dict, ephemeral: bool = False):
        return Message(source=SourceType.OPENHANDS, message=msg, ephemeral=ephemeral)
