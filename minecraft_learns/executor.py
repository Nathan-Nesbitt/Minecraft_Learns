"""
"""

from .errors import TaskNotFound
from tasks.animal_use import AnimalUseTask
from tasks.determine_build import DetermineBuildTask
from tasks.identify_danger import IdentifyDangerTask
from tasks.identify_npc import IdentifyNPCTask
from tasks.predict_farm import PredictFarmTask


class Executor:
    """
    Control class for the minecraft_learns system
    """
    def __init__(self, uuid):
        self.uuid=uuid

    def set_task(self, task_name):
        """
        set the goal to the name given
        ---
        @param task_name: string name of the task to do
        """        
        task_map = {
            "animal_use": AnimalUseTask(),
            "determine_build.py": DetermineBuildTask(),
            "identify_danger.py": IdentifyDangerTask(),
            "identify_npc": IdentifyNPCTask(),
            "predict_farm": PredictFarmTask(),
            "predict_mine": PredictMineTask()
        }

        if not task_name in task_map.keys():
            raise TaskNotFound(task_name)
        
        self.task = task_map[task_name]        



