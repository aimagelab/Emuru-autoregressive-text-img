class TrainState:
    """
    This class is used to store the state of the training process.
    """

    def __init__(self, global_step: int, epoch: int, best_eval_init: float):
        """
        Initialize the TrainState object. 
        Args:
            global_step: The global step of the training process.
            epoch: The epoch of the training process.
            best_eval_init: The initial best evaluation score.
        """
        self.global_step = global_step
        self.epoch = epoch
        self.best_eval = best_eval_init
        self.last_eval = best_eval_init

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrainState):
            return self.global_step == other.global_step and self.epoch == other.epoch and self.best_eval == other.best_eval and self.last_eval == other.last_eval
        return False

    def __repr__(self) -> str:
        return f"TrainState(global_step={self.global_step}, epoch={self.epoch}, best_eval={self.best_eval}, last_eval={self.last_eval})"

    def state_dict(self) -> dict:
        return {'global_step': self.global_step, 'epoch': self.epoch, 'best_eval': self.best_eval, 'last_eval': self.last_eval}

    def load_state_dict(self, state_dict: dict) -> None:
        self.global_step = state_dict['global_step']
        self.epoch = state_dict['epoch']
        self.best_eval = state_dict.get('best_eval', 0.0)
        self.last_eval = state_dict.get('last_eval', 0.0)
