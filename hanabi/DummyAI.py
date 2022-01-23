from BaseAI import *

# Basic 'Dumb' AI, simply doing random moves
class DummyAI(BaseAI):
    def __init__(self, show_moves: List[ShowMoves] = None, save_game_states=False):
        super().__init__(show_moves=show_moves, save_game_states=save_game_states)

    # Override generic event to make the random move
    def generic_event(self, data):
        super().generic_event(data)
        if self.current_player == self.client.player_name and self.ready_to_play and self.is_updated:
            self.random_move()
