
from fytok.utils.base import IDS, FyContext

from fytok.ontology import pulse_schedule


class PulseSchedule(IDS, FyContext, pulse_schedule.pulse_schedule):
    pass
