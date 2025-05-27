import gymnasium as gym
from gymnasium import spaces, utils
import numpy as np
import pygame
from io import StringIO
from contextlib import closing
import os
from PIL import Image

MOVE_LEFT = 0
MOVE_DOWN = 1
MOVE_RIGHT = 2
MOVE_UP = 3

MAPS = {
    "map1": [
        "SFFFFHFH2FFHFFFFHFFF",
        "FFFHFFHHHHFFFHHFFFHF",
        "FHFHFHFHFHHHHHHHHHHF",
        "FHFFFHFAFHFFFHFFFHFF",
        "FHFFHHFHFHFHFHHHFHFH",
        "HHHFFHFHFHFHFFFHFHFH",
        "FHHHFHFHFHFHFHFHFHFH",
        "FFFHFFFHFFFHFFFHFFFH",
        "FHFHFHFHFHFHFHFHFHFH",
        "FHFFFHFHFHFFFHFFFHFH",
        "FHHHFHHHFHHHFHHHFFFH",
        "FFFHFFFHFFFHFFFHFHFH",
        "FHFHFHFHFHFHFHFHFHFH",
        "FHFFFHFHFHFFFHFFFHFH",
        "FHFHFHFHFHFHFHFHFHFH",
        "FFFHFFFHFHFFFHFHFHHH",
        "HHFHFHFHFHFHFHFHHHFH",
        "HHHHFHFHFHFFFHFFFBFH",
        "1FFHFHFHFHFHFHFHFHFH",
        "HHFFFHFHFHFHFHFHFHGH"
    ]
}


class KeyDoorEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, desc=None, map_name="map1",
                 goal_reward=200.0, key_pickup_reward=30.0, door_open_reward=50.0,
                 hole_penalty=-100.0, move_cost=-1.0, bump_penalty=-10.0,
                 interactive_mode=False):
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = os.path.join(os.path.dirname(self.working_dir), 'assets')

        if desc is None:
            map_desc_array = MAPS.get(map_name, MAPS["map1"])
        else:
            map_desc_array = desc

        self.desc_np = np.asarray(map_desc_array, dtype="c")
        self.nrow, self.ncol = self.desc_np.shape
        self.current_map_state = np.copy(self.desc_np)

        self.goal_reward = goal_reward
        self.key_pickup_reward = key_pickup_reward
        self.door_open_reward = door_open_reward
        self.hole_penalty = hole_penalty
        self.move_cost = move_cost
        self.bump_penalty = bump_penalty

        self.key1_collected = False
        self.key2_collected = False
        self.door1_unlocked = False
        self.door2_unlocked = False

        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.nrow * self.ncol)

        start_coords = np.argwhere(self.desc_np == b'S')
        self.start_row, self.start_col = start_coords[0]
        self.agent_row = 0
        self.agent_col = 0
        self.state = 0

        self.render_mode = render_mode
        self.interactive_mode = interactive_mode
        self.tile_pixel_size = 32
        self.fullscreen_active = False
        self.scale_factor = 1.0
        self.min_scale = 0.25
        self.max_scale = 4.0
        self.scale_increment = 0.1
        self.camera_follows_player = True
        self.target_fps = self.metadata["render_fps"]
        self.fps_min = 5
        self.fps_max = 120
        self.fps_increment = 5
        self.cam_offset_x = 0
        self.cam_offset_y = 0
        self.default_win_width = 850
        self.default_win_height = 600
        self.game_panel_ratio = 0.7
        self.info_panel_ratio = 0.3
        self.display_width = 0
        self.display_height = 0

        self._compute_window_dimensions()

        self.display_surface = None
        self.timer_clock = None
        self.primary_font = None
        self.secondary_font = None
        self.texture_cache = {}
        self.pygame_ready = False
        self.graphics_loaded = False
        self.source_textures = {}
        self.active_tile_size = 0

        self.asset_filenames = {
            'S': "start_tile.png", 'G': "goal.png", 'F': "floor.png", 'H': "wall.png",
            '1': "key_gold.png", '2': "key_silver.png", 'A': "door_gold_closed.png",
            'B': "door_silver_closed.png", 'X': "door_gold_open.png", 'Y': "door_silver_open.png",
            'AGENT': "player.png"
        }

        self.tile_color_map = {
            b'S': (100, 220, 100), b'G': (255, 215, 0), b'F': (200, 200, 210), b'H': (60, 60, 80),
            b'1': (255, 190, 0), b'2': (192, 192, 192), b'A': (139, 69, 19), b'B': (120, 120, 120),
            b'X': (205, 133, 63), b'Y': (160, 160, 160)
        }

        self.player_color = (255, 80, 80)
        self.movement_trail = []
        self.awaiting_reset = False
        self.previous_action = None
        self.step_info_data = {}

        if self.render_mode == "human":
            self._setup_pygame()

        self.reset()

    def _compute_window_dimensions(self):
        if self.fullscreen_active and pygame.get_init() and pygame.display.get_init():
            screen_info = pygame.display.Info()
            self.display_width, self.display_height = screen_info.current_w, screen_info.current_h
        else:
            self.display_width, self.display_height = self.default_win_width, self.default_win_height
        self.game_panel_width = int(self.display_width * self.game_panel_ratio)
        self.game_panel_height = self.display_height
        self.info_panel_width = self.display_width - self.game_panel_width

    def _setup_pygame(self):
        if not self.pygame_ready:
            if not pygame.get_init():
                pygame.init()
            if not pygame.font.get_init():
                pygame.font.init()
            if not pygame.display.get_init():
                pygame.display.init()
            if self.display_width == 0 or self.display_height == 0:
                self._compute_window_dimensions()

            if self.fullscreen_active:
                self.display_surface = pygame.display.set_mode((self.display_width, self.display_height),
                                                      pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self.display_surface = pygame.display.set_mode((self.display_width, self.display_height), pygame.RESIZABLE)
            pygame.display.set_caption(f"KeyDoorEnv - {self.nrow}x{self.ncol} - 2 Keys")

            self.primary_font = pygame.font.Font(None, 24)
            self.secondary_font = pygame.font.Font(None, 20)
            if self.timer_clock is None:
                self.timer_clock = pygame.time.Clock()
            self.graphics_loaded = False
            self.pygame_ready = True

    def _prepare_assets(self):
        render_size = self.tile_pixel_size
        if self.render_mode == "human" and self.pygame_ready and self.game_panel_width > 0 and self.ncol > 0 and self.game_panel_height > 0 and self.nrow > 0:
            width_per_tile = self.game_panel_width / self.ncol
            height_per_tile = self.game_panel_height / self.nrow
            computed_size = int(min(width_per_tile, height_per_tile) * self.scale_factor)
            if computed_size > 0:
                render_size = computed_size
        self.texture_cache = {}
        self.source_textures = {}
        any_loaded = False
        if os.path.isdir(self.resource_path):
            for sprite_key, file_name in self.asset_filenames.items():
                full_path = os.path.join(self.resource_path, file_name)
                if os.path.exists(full_path):
                    pil_image = Image.open(full_path).convert("RGBA")
                    self.source_textures[sprite_key] = pil_image
                    pygame_surf = pygame.image.fromstring(pil_image.tobytes(), pil_image.size,
                                                             pil_image.mode).convert_alpha()
                    self.texture_cache[sprite_key] = pygame.transform.scale(pygame_surf, (render_size, render_size))
                    any_loaded = True
                else:
                    self.texture_cache[sprite_key] = None
        if any_loaded:
            self.graphics_loaded = True
            self.active_tile_size = render_size
        else:
            self.graphics_loaded = False

    def _switch_fullscreen(self):
        self.fullscreen_active = not self.fullscreen_active
        pygame.display.quit()
        pygame.display.init()
        self._setup_pygame()
        self.graphics_loaded = False

    def _modify_scale(self, direction):
        if direction > 0:
            self.scale_factor = min(self.max_scale, self.scale_factor + self.scale_increment)
        elif direction < 0:
            self.scale_factor = max(self.min_scale, self.scale_factor - self.scale_increment)
        self.graphics_loaded = False

    def _modify_framerate(self, direction):
        if direction > 0:
            self.target_fps = min(self.fps_max, self.target_fps + self.fps_increment)
        elif direction < 0:
            self.target_fps = max(self.fps_min, self.target_fps - self.fps_increment)

    def _process_input_events(self):
        if self.render_mode != "human" or not pygame.display.get_init():
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                import sys
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._switch_fullscreen()
                elif event.key in [pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS]:
                    self._modify_scale(1)
                elif event.key in [pygame.K_MINUS, pygame.K_KP_MINUS]:
                    self._modify_scale(-1)
                elif event.key == pygame.K_c:
                    self.camera_follows_player = not self.camera_follows_player
                elif event.key == pygame.K_PAGEUP:
                    self._modify_framerate(1)
                elif event.key == pygame.K_PAGEDOWN:
                    self._modify_framerate(-1)
                elif event.key == pygame.K_r:
                    if self.interactive_mode and self.awaiting_reset:
                        self.reset()
            if event.type == pygame.VIDEORESIZE and not self.fullscreen_active:
                self.display_width = event.w
                self.display_height = event.h
                self._compute_window_dimensions()
                window_title = pygame.display.get_caption()[0]
                self.display_surface = pygame.display.set_mode((self.display_width, self.display_height), pygame.RESIZABLE)
                pygame.display.set_caption(window_title)
                self.graphics_loaded = False

    def _coord_to_state(self, row, col):
        return row * self.ncol + col

    def _collect_game_data(self):
        return {
            "prob": 1.0,
            "has_key1": self.key1_collected,
            "door1_open": self.door1_unlocked,
            "has_key2": self.key2_collected,
            "door2_open": self.door2_unlocked
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_row = self.start_row
        self.agent_col = self.start_col
        self.state = self._coord_to_state(self.agent_row, self.agent_col)
        self.key1_collected = False
        self.key2_collected = False
        self.door1_unlocked = False
        self.door2_unlocked = False
        self.current_map_state = np.copy(self.desc_np)
        self.previous_action = None
        self.step_info_data = {}
        self.cam_offset_x = 0
        self.cam_offset_y = 0
        self.movement_trail = [(self.agent_row, self.agent_col)]
        self.awaiting_reset = False

        if self.render_mode == "human":
            self._setup_pygame()
            if self.pygame_ready and self.display_surface:
                self._draw_game()

        observation = self.state
        info = self._collect_game_data()
        return observation, info

    def step(self, action):
        if self.interactive_mode and self.awaiting_reset:
            return self.state, 0.0, True, False, self._collect_game_data()

        self.previous_action = action
        reward = self.move_cost
        terminated = False
        truncated = False
        action_info = {}

        old_row, old_col = self.agent_row, self.agent_col
        target_row, target_col = self.agent_row, self.agent_col

        if action == MOVE_LEFT:
            target_col -= 1
        elif action == MOVE_DOWN:
            target_row += 1
        elif action == MOVE_RIGHT:
            target_col += 1
        elif action == MOVE_UP:
            target_row -= 1

        if 0 <= target_row < self.nrow and 0 <= target_col < self.ncol:
            self.agent_row = target_row
            self.agent_col = target_col
            if not self.movement_trail or self.movement_trail[-1] != (self.agent_row, self.agent_col):
                self.movement_trail.append((self.agent_row, self.agent_col))
        else:
            reward += self.bump_penalty
            action_info["message"] = "Hit boundary!"

        tile_at_position = self.current_map_state[self.agent_row, self.agent_col]

        if tile_at_position == b'H':
            reward += self.hole_penalty
            terminated = True
            action_info["message"] = "Trap!"
        elif tile_at_position == b'1':
            if not self.key1_collected:
                self.key1_collected = True
                reward += self.key_pickup_reward
                self.current_map_state[self.agent_row, self.agent_col] = b'F'
                action_info["message"] = "Gold Key!"
        elif tile_at_position == b'2':
            if not self.key2_collected:
                self.key2_collected = True
                reward += self.key_pickup_reward
                self.current_map_state[self.agent_row, self.agent_col] = b'F'
                action_info["message"] = "Silver Key!"
        elif tile_at_position == b'A':
            if self.key1_collected and not self.door1_unlocked:
                self.door1_unlocked = True
                reward += self.door_open_reward
                self.current_map_state[self.agent_row, self.agent_col] = b'X'
                action_info["message"] = "Gold Door Opened!"
            elif not self.key1_collected:
                reward += self.bump_penalty
                self.agent_row = old_row
                self.agent_col = old_col
                action_info["message"] = "Need Gold Key!"
        elif tile_at_position == b'B':
            if self.key2_collected and not self.door2_unlocked:
                self.door2_unlocked = True
                reward += self.door_open_reward
                self.current_map_state[self.agent_row, self.agent_col] = b'Y'
                action_info["message"] = "Silver Door Opened!"
            elif not self.key2_collected:
                reward += self.bump_penalty
                self.agent_row = old_row
                self.agent_col = old_col
                action_info["message"] = "Need Silver Key!"
        elif tile_at_position == b'G':
            if self.door1_unlocked and self.door2_unlocked:
                reward += self.goal_reward
                terminated = True
                action_info["message"] = "Goal Reached!!!"
            else:
                action_info["message"] = "Goal locked!"

        self.state = self._coord_to_state(self.agent_row, self.agent_col)
        self.step_info_data = action_info

        complete_info = self._collect_game_data()
        complete_info.update(action_info)

        if terminated and self.interactive_mode:
            self.awaiting_reset = True

        if self.render_mode == "human":
            self._draw_game()

        return self.state, reward, terminated, truncated, complete_info

    def _draw_game(self):
        if self.render_mode != "human":
            return
        self._setup_pygame()
        if self.display_surface is None:
            return
        self._process_input_events()
        if self.display_surface is None:
            return

        current_tile_size = self.tile_pixel_size
        if self.pygame_ready and self.game_panel_width > 0 and self.ncol > 0 and self.game_panel_height > 0 and self.nrow > 0:
            tile_width = self.game_panel_width / self.ncol
            tile_height = self.game_panel_height / self.nrow
            current_tile_size = int(min(tile_width, tile_height) * self.scale_factor)
            current_tile_size = max(1, current_tile_size)

        if not self.graphics_loaded or self.active_tile_size != current_tile_size:
            self._prepare_assets()

        actual_tile_size = self.active_tile_size if self.graphics_loaded and self.active_tile_size > 0 else current_tile_size
        main_canvas = pygame.Surface((self.display_width, self.display_height))
        main_canvas.fill((50, 50, 80))
        game_canvas = pygame.Surface((self.game_panel_width, self.game_panel_height))
        game_canvas.fill((100, 140, 180))

        viewport_x, viewport_y = 0, 0
        if self.ncol > 0 and self.nrow > 0:
            total_map_width = self.ncol * actual_tile_size
            total_map_height = self.nrow * actual_tile_size
            player_x = self.agent_col * actual_tile_size
            player_y = self.agent_row * actual_tile_size

            if self.camera_follows_player:
                viewport_x = (self.game_panel_width / 2) - (player_x + actual_tile_size / 2)
                viewport_y = (self.game_panel_height / 2) - (player_y + actual_tile_size / 2)
                if total_map_width > self.game_panel_width:
                    viewport_x = max(self.game_panel_width - total_map_width, min(0, viewport_x))
                else:
                    viewport_x = (self.game_panel_width - total_map_width) / 2
                if total_map_height > self.game_panel_height:
                    viewport_y = max(self.game_panel_height - total_map_height, min(0, viewport_y))
                else:
                    viewport_y = (self.game_panel_height - total_map_height) / 2
            else:
                if total_map_width <= self.game_panel_width:
                    viewport_x = (self.game_panel_width - total_map_width) / 2
                else:
                    viewport_x = max(self.game_panel_width - total_map_width, min(0, self.cam_offset_x))
                if total_map_height <= self.game_panel_height:
                    viewport_y = (self.game_panel_height - total_map_height) / 2
                else:
                    viewport_y = max(self.game_panel_height - total_map_height, min(0, self.cam_offset_y))

        for row_idx in range(self.nrow):
            for col_idx in range(self.ncol):
                draw_x = col_idx * actual_tile_size + viewport_x
                draw_y = row_idx * actual_tile_size + viewport_y
                if not (
                        draw_x + actual_tile_size < 0 or draw_x > self.game_panel_width or draw_y + actual_tile_size < 0 or draw_y > self.game_panel_height):
                    tile_rect = pygame.Rect(draw_x, draw_y, actual_tile_size, actual_tile_size)
                    cell_byte = self.current_map_state[row_idx, col_idx]
                    cell_char = cell_byte.decode('utf-8')
                    texture = self.texture_cache.get(cell_char)

                    if texture:
                        game_canvas.blit(texture, tile_rect)
                    else:
                        tile_color = self.tile_color_map.get(cell_byte, (128, 128, 128))
                        pygame.draw.rect(game_canvas, tile_color, tile_rect)
                    if actual_tile_size > 2:
                        pygame.draw.rect(game_canvas, (30, 30, 40, 150), tile_rect, 1)

        player_draw_x = self.agent_col * actual_tile_size + viewport_x
        player_draw_y = self.agent_row * actual_tile_size + viewport_y
        player_rect = pygame.Rect(player_draw_x, player_draw_y, actual_tile_size, actual_tile_size)
        player_texture = self.texture_cache.get('AGENT')
        if player_texture:
            game_canvas.blit(player_texture, player_rect)
        else:
            pygame.draw.rect(game_canvas, self.player_color,
                             player_rect.inflate(-actual_tile_size // 4, -actual_tile_size // 4))

        main_canvas.blit(game_canvas, (0, 0))

        if self.primary_font:
            panel_bg = (50, 60, 70)
            pygame.draw.rect(main_canvas, panel_bg, (self.game_panel_width, 0, self.info_panel_width, self.display_height))
            info_start_x = self.game_panel_width + 15
            current_y = 20
            text_shade = (230, 230, 240)

            def render_text_line(surface, text, x, y, font, color):
                text_surf = font.render(text, True, color)
                surface.blit(text_surf, (x, y))
                return y + font.get_height() + 4

            k1_shade = (255, 223, 0) if self.key1_collected else text_shade
            current_y = render_text_line(main_canvas, f"K1:{'YES' if self.key1_collected else 'NO'}", info_start_x, current_y, self.primary_font,
                               k1_shade)
            d1_shade = (160, 255, 160) if self.door1_unlocked else text_shade
            current_y = render_text_line(main_canvas, f"D1:{'OPEN' if self.door1_unlocked else 'CLOSED'}", info_start_x, current_y, self.primary_font,
                               d1_shade)
            k2_shade = (210, 210, 220) if self.key2_collected else text_shade
            current_y = render_text_line(main_canvas, f"K2:{'YES' if self.key2_collected else 'NO'}", info_start_x, current_y, self.primary_font,
                               k2_shade)
            d2_shade = (160, 255, 160) if self.door2_unlocked else text_shade
            current_y = render_text_line(main_canvas, f"D2:{'OPEN' if self.door2_unlocked else 'CLOSED'}", info_start_x, current_y, self.primary_font,
                               d2_shade)
            current_y += 10

            if self.previous_action is not None:
                current_y = render_text_line(main_canvas, f"Action:{['L', 'D', 'R', 'U'][self.previous_action]}", info_start_x, current_y,
                                   self.secondary_font, text_shade)

            if "message" in self.step_info_data and self.step_info_data["message"]:
                message_text = self.step_info_data["message"]
                message_shade = text_shade
                if "Goal" in message_text or "Opened" in message_text or "Key!" in message_text:
                    message_shade = (100, 255, 100)
                elif "Need" in message_text or "Hit" in message_text or "Trap" in message_text:
                    message_shade = (255, 100, 100)

                text_lines = [line.split(' ') for line in message_text.splitlines()]
                panel_max_width = self.info_panel_width - 30
                for word_group in text_lines:
                    line_buffer = ""
                    for word in word_group:
                        if self.secondary_font.size(line_buffer + word + " ")[0] < panel_max_width:
                            line_buffer += word + " "
                        else:
                            current_y = render_text_line(main_canvas, line_buffer.strip(), info_start_x, current_y, self.secondary_font, message_shade)
                            line_buffer = word + " "
                    if line_buffer.strip():
                        current_y = render_text_line(main_canvas, line_buffer.strip(), info_start_x, current_y, self.secondary_font, message_shade)

        if self.display_surface:
            self.display_surface.blit(main_canvas, main_canvas.get_rect())
            pygame.display.update()
        if self.timer_clock:
            self.timer_clock.tick(self.target_fps)

    def _generate_text_view(self):
        output = StringIO()
        agent_r, agent_c = self.agent_row, self.agent_col
        text_grid = [[char.decode() for char in row] for row in self.current_map_state.tolist()]
        text_grid[agent_r][agent_c] = f"X({text_grid[agent_r][agent_c]})"

        status_line = f"K1:{'T' if self.key1_collected else 'F'},D1:{'O' if self.door1_unlocked else 'C'}|K2:{'T' if self.key2_collected else 'F'},D2:{'O' if self.door2_unlocked else 'C'}"
        if self.previous_action is not None:
            status_line += f"|A:{['L', 'D', 'R', 'U'][self.previous_action]}"
        if "message" in self.step_info_data:
            status_line += f"|M:{self.step_info_data['message']}"

        output.write(status_line + "\n" + "\n".join("".join(line) for line in text_grid) + "\n")
        return output.getvalue()

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == "human":
            self._setup_pygame()
            self._draw_game()
        elif self.render_mode == "ansi":
            print(self._generate_text_view())
        elif self.render_mode == "rgb_array":
            if not pygame.get_init():
                pygame.init()
            self._setup_pygame()
            tile_size = self.tile_pixel_size
            canvas = pygame.Surface((self.ncol * tile_size, self.nrow * tile_size))
            canvas.fill((40, 40, 60))
            for row_idx in range(self.nrow):
                for col_idx in range(self.ncol):
                    tile_rect = pygame.Rect(col_idx * tile_size, row_idx * tile_size, tile_size, tile_size)
                    tile_byte = self.current_map_state[row_idx, col_idx]
                    tile_color = (160, 170, 180)
                    if tile_byte == b'S':
                        tile_color = (100, 220, 100)
                    elif tile_byte == b'F':
                        tile_color = (200, 200, 210)
                    elif tile_byte == b'H':
                        tile_color = (60, 60, 80)
                    elif tile_byte == b'G':
                        tile_color = (255, 215, 0)
                    elif tile_byte == b'1':
                        tile_color = (255, 190, 0)
                    elif tile_byte == b'2':
                        tile_color = (192, 192, 192)
                    elif tile_byte == b'A':
                        tile_color = (139, 69, 19)
                    elif tile_byte == b'B':
                        tile_color = (120, 120, 120)
                    elif tile_byte == b'X':
                        tile_color = (205, 133, 63)
                    elif tile_byte == b'Y':
                        tile_color = (160, 160, 160)
                    pygame.draw.rect(canvas, tile_color, tile_rect)
                    if tile_size > 4:
                        pygame.draw.rect(canvas, (30, 30, 40), tile_rect, 1)
            pygame.draw.rect(canvas, (255, 80, 80), pygame.Rect(self.agent_col * tile_size, self.agent_row * tile_size, tile_size, tile_size))
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.pygame_ready and pygame.get_init():
            if self.display_surface is not None:
                pygame.display.quit()
            pygame.font.quit()
            pygame.quit()
        self.display_surface = None
        self.timer_clock = None
        self.primary_font = None
        self.secondary_font = None
        self.texture_cache = {}
        self.graphics_loaded = False
        self.source_textures = {}
        self.pygame_ready = False
