# Snake_env.py

import random
import pygame
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- CONFIGURATION ---
HEIGHT_SPACE = 450
SCORE_SPACE = 50
HEIGHT = HEIGHT_SPACE + SCORE_SPACE
WIDTH = 600
BLOCK_SIZE = 30
RENDER_FPS = 30 # Tăng lên 30 cho mượt

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (30, 30, 30)

class Food:
    def __init__(self):
        self.position = (0, 0)
        # Random vị trí ngay khi khởi tạo
        self.respawn([])

    def respawn(self, snake_body):
        # Đảm bảo food không spawn trùng vào thân rắn
        while True:
            x = random.randrange(0, WIDTH, BLOCK_SIZE)
            y = random.randrange(SCORE_SPACE, HEIGHT, BLOCK_SIZE)
            if (x, y) not in snake_body:
                self.position = (x, y)
                break

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))


class Snake:
    def __init__(self):
        start_x = WIDTH // 2 // BLOCK_SIZE * BLOCK_SIZE
        start_y = SCORE_SPACE + HEIGHT // 2 // BLOCK_SIZE * BLOCK_SIZE
        self.body = [(start_x, start_y), (start_x - BLOCK_SIZE, start_y), (start_x - 2 * BLOCK_SIZE, start_y)]
        self.direction = (BLOCK_SIZE, 0) # Mặc định đi sang phải
        self.growing_pending = 0

    def head(self):
        return self.body[0]

    def move(self, action):
        # Action: 0=Left, 1=Right, 2=Up, 3=Down
        cur_dx, cur_dy = self.direction

        if action == 0:   # LEFT
            new_dir = (-BLOCK_SIZE, 0)
        elif action == 1: # RIGHT
            new_dir = (BLOCK_SIZE, 0)
        elif action == 2: # UP
            new_dir = (0, -BLOCK_SIZE)
        elif action == 3: # DOWN
            new_dir = (0, BLOCK_SIZE)
        else:
            new_dir = self.direction

        # Ngăn rắn quay đầu 180 độ (tự cắn cổ)
        if (new_dir[0] == -cur_dx and new_dir[1] == -cur_dy):
            new_dir = self.direction

        self.direction = new_dir
        
        # Cập nhật vị trí đầu mới
        head_x, head_y = self.body[0]
        new_head = (head_x + new_dir[0], head_y + new_dir[1])

        self.body.insert(0, new_head)

        # Xử lý việc dài ra hoặc giữ nguyên độ dài
        if self.growing_pending > 0:
            self.growing_pending -= 1
        else:
            self.body.pop()

    def grow(self, amount=1):
        self.growing_pending += amount

    def check_collision(self):
        head_x, head_y = self.head()

        # Va chạm tường
        if head_x < 0 or head_x >= WIDTH or head_y < SCORE_SPACE or head_y >= HEIGHT:
            return True

        # Va chạm thân mình (bỏ qua đầu)
        if self.head() in self.body[1:]:
            return True

        return False

    def draw(self, surface):
        for i, (x, y) in enumerate(self.body):
            color = GREEN if i == 0 else BLUE
            pygame.draw.rect(surface, color, (x, y, BLOCK_SIZE, BLOCK_SIZE))


def draw_grid(surface):
    # Vẽ lưới mờ để dễ nhìn
    for x in range(0, WIDTH, BLOCK_SIZE):
        pygame.draw.line(surface, GRAY, (x, SCORE_SPACE), (x, HEIGHT), 1)
    for y in range(SCORE_SPACE, HEIGHT, BLOCK_SIZE):
        pygame.draw.line(surface, GRAY, (0, y), (WIDTH, y), 1)
    # Vẽ đường phân cách Score
    pygame.draw.line(surface, WHITE, (0, SCORE_SPACE), (WIDTH, SCORE_SPACE), 2)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        
        self.render_mode = render_mode
        self.width = WIDTH
        self.height = HEIGHT
        
        # Define Action Space: 4 hướng
        self.action_space = spaces.Discrete(4)

        # Define Observation Space: 8 giá trị float (như trong _get_state của bạn)
        # Các giá trị normalized khoảng -1 đến 1, hoặc 0 đến 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.window = None
        self.clock = None
        self.font = None

        # Khởi tạo đối tượng game
        self.snake = None
        self.food = None
        self.score = 0
        self.prev_distance = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Khởi tạo lại game
        self.snake = Snake()
        self.food = Food()
        
        # Đảm bảo thức ăn không spawn trùng rắn lúc đầu
        self.food.respawn(self.snake.body)
        
        self.score = 0
        self.prev_distance = self._manhattan_distance(self.snake.head(), self.food.position)
        
        # Render frame đầu tiên nếu cần
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_state(), {}

    def step(self, action):
        # Thực hiện hành động
        self.snake.move(action)
        
        terminated = False
        truncated = False # Dùng nếu muốn giới hạn số bước đi tối đa
        reward = 0.0

        # 1. Ăn mồi
        if self.snake.head() == self.food.position:
            self.score += 1
            self.snake.grow(1)
            self.food.respawn(self.snake.body)
            reward += 10.0
            # Reset distance để tránh reward cộng dồn sai logic
            self.prev_distance = self._manhattan_distance(self.snake.head(), self.food.position)
        
        # 2. Va chạm (Chết)
        elif self.snake.check_collision():
            reward -= 10.0
            terminated = True
        
        # 3. Di chuyển thông thường (Shaping Reward + Time Penalty)
        else:
            # A. PHẠT THỜI GIAN (Step Penalty)
            # reward -= 0.01 
            
            new_distance = self._manhattan_distance(self.snake.head(), self.food.position)
            
            # B. SHAPING REWARD (Thưởng nhỏ, Phạt nặng)
            if new_distance < self.prev_distance:
                # Lại gần: Thưởng cực nhỏ (chỉ để dẫn đường)
                reward += 0.02
            # else:
                # Đi xa ra: Phạt nặng hơn thưởng (để tránh đi vòng vèo)
                # reward -= 0.1 
            
            self.prev_distance = new_distance

        # Render nếu cần
        if self.render_mode == "human":
            self._render_frame()

        return self._get_state(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Snake RL Env")
            else:
                self.window = pygame.Surface((WIDTH, HEIGHT))
            
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)

        # Vẽ nền
        self.window.fill(BLACK)
        draw_grid(self.window)

        # Vẽ các đối tượng
        if self.food:
            self.food.draw(self.window)
        if self.snake:
            self.snake.draw(self.window)

        # Vẽ điểm số
        if self.font:
            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.window.blit(score_text, (10, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        # Trả về numpy array cho mode rgb_array
        if self.render_mode == "rgb_array":
            texture = self.window
            array = pygame.surfarray.array3d(texture)
            return np.transpose(array, (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _get_state(self):
        # Trả về numpy array thay vì tensor (chuẩn Gym)
        head_x, head_y = self.snake.head()
        food_x, food_y = self.food.position
        
        # Normalize inputs
        dx = (food_x - head_x) / WIDTH
        dy = (food_y - head_y) / HEIGHT
        
        dir_x = self.snake.direction[0] / BLOCK_SIZE
        dir_y = self.snake.direction[1] / BLOCK_SIZE

        # Check danger (Local view)
        point_l = (head_x - BLOCK_SIZE, head_y)
        point_r = (head_x + BLOCK_SIZE, head_y)
        point_u = (head_x, head_y - BLOCK_SIZE)
        point_d = (head_x, head_y + BLOCK_SIZE)

        def is_danger(pt):
            x, y = pt
            # Hit wall
            if x < 0 or x >= WIDTH or y < SCORE_SPACE or y >= HEIGHT:
                return 1.0
            # Hit self
            if pt in self.snake.body[1:]:
                return 1.0
            return 0.0

        state = np.array([
            dx, dy,          # Relative Food Position
            dir_x, dir_y,    # Current Direction
            is_danger(point_u), # Danger Up
            is_danger(point_d), # Danger Down
            is_danger(point_l), # Danger Left
            is_danger(point_r)  # Danger Right
        ], dtype=np.float32)

        return state

    def _manhattan_distance(self, head, food):
        return abs(head[0] - food[0]) + abs(head[1] - food[1])

if __name__ == "__main__":
    # Test nhanh xem môi trường chạy ổn không
    env = SnakeEnv(render_mode="human")
    obs, _ = env.reset()
    print("Shape obs:", obs.shape)
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
    env.close()


import gymnasium as gym
from gymnasium.envs.registration import register

ENV_ID = "Snake-v1"

# Xóa đăng ký cũ nếu trùng (để reload module không bị lỗi)
if ENV_ID in gym.envs.registry:
    del gym.envs.registry[ENV_ID]

register(
    id=ENV_ID,
    # Lưu ý: Vì code đang nằm trong chính file này nên dùng __name__
    # hoặc dùng "Snake_env:SnakeEnv" nếu file tên là Snake_env.py
    entry_point="Snake_env:SnakeEnv", 
    max_episode_steps=2000,
)