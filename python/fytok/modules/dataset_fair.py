import getpass
import os
import datetime
from spdm.core.sp_property import sp_tree, sp_property

from ..ontology import dataset_fair, GLOBAL_ONTOLOGY


@sp_tree
class DataDescription:
    device: str

    shot: int

    run: int = sp_property(default_value=0)

    summary: str = sp_property(default_value="")

    def __str__(self) -> str:
        return f"{self.device.upper()} #{self.shot}/{self.run}"

    @sp_property
    def tag(self) -> str:
        return f"{self.device.lower()}_{self.shot}_{self.run}"


@sp_tree
class DatasetFAIR(dataset_fair._T_dataset_fair):
    ontology: str = GLOBAL_ONTOLOGY

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
