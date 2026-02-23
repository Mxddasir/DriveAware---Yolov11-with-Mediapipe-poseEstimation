from typing import Tuple


class DecisionEngine:
    # decides if the driver is using their phone based on detection + pose signals

    @staticmethod
    def decide_phone_use(
        found_high: bool,
        found_low: bool,
        suspicious: bool,
        hand_ok: bool,
        require_hand_proximity: bool
    ) -> Tuple[bool, str]:

        # high confidence detection
        if found_high:
            # if we need hand proximity, check hand is near or pose is suspicious
            if require_hand_proximity and (not hand_ok) and (not suspicious):
                return False, "HighConf phone but NOT near hand/face (filtered)"
            return True, "HighConf phone"

        # low confidence only counts if pose looks suspicious too
        if suspicious and found_low:
            if require_hand_proximity and (not hand_ok):
                return False, "LowConf phone + suspicious, but NOT near hand (filtered)"
            return True, "LowConf phone + suspicious"

        return False, "No phone (or not suspicious)"