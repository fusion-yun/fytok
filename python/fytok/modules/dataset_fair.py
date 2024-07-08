import getpass
import os
import datetime
from spdm.core.sp_tree import sp_tree, sp_property


from fytok import ontology


@sp_tree
class DataDescription:
    device: str

    shot: int

    run: int = 0

    summary: str = ""

    def __str__(self) -> str:
        return f"{self.device.upper()} #{self.shot}/{self.run}"

    @sp_property
    def tag(self) -> str:
        return f"{self.device.lower()}_{self.shot}_{self.run}"


class DatasetFAIR(ontology.dataset_fair.dataset_fair):
    ontology: str = ontology.__VERSION__

    description: DataDescription

    @sp_property
    def creator(self) -> str:
        return getpass.getuser().capitalize()

    @sp_property
    def create_time(self) -> str:
        return datetime.datetime.now().isoformat()

    @sp_property
    def site(self) -> str:
        return os.uname().nodename

    def __str__(self) -> str:
        return f""" 
    Device: {self.description.device.upper()}, Shot: {self.description.shot}, Run: {self.description.run}, 
    Run by {self.creator} on {self.site} at {self.create_time}, base on ontology \"{self.ontology}\"
"""
