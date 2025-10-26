# grid_moving_points.py
# - 2×5 = 10개의 정사각형
# - 각 박스: 중심 고정점 + 움직이는 점들
# - 중심을 기준으로 Q1~Q4 사분면 개수를 실시간 카운트
# 필요 패키지: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

# ====== 파라미터 ======
ROWS, COLS = 2, 5          # 2×5 = 10개
BOX_SIZE = 1.0             # 정사각형 한 변 길이
GAP = 0.25                 # 박스 간 간격
POINTS_PER_BOX = 20        # 박스별 이동 점 개수
SPEED = 0.02               # 점 속도(프레임당)
FRAMES = 300               # 프레임 수
INTERVAL = 30              # 프레임 간(ms)

# ====== 레이아웃 ======
width = COLS * BOX_SIZE + (COLS - 1) * GAP
height = ROWS * BOX_SIZE + (ROWS - 1) * GAP

# 각 박스의 좌하단 좌표 (x0,y0)
origins = []
for r in range(ROWS):
    for c in range(COLS):
        x0 = c * (BOX_SIZE + GAP)
        y0 = (ROWS - 1 - r) * (BOX_SIZE + GAP)  # 맨 위 행이 큰 y
        origins.append((x0, y0))

# ====== 상태 초기화 ======
rng = np.random.default_rng(42)
positions = []   # 각 박스 내 로컬 좌표 [0, BOX_SIZE]
velocities = []
for _ in range(ROWS * COLS):
    # 테두리와 너무 붙지 않게 0.05 마진
    p = rng.uniform(0.05, BOX_SIZE - 0.05, size=(POINTS_PER_BOX, 2))
    theta = rng.uniform(0, 2*np.pi, size=(POINTS_PER_BOX,))
    v = np.stack([np.cos(theta), np.sin(theta)], axis=1) * SPEED
    positions.append(p)
    velocities.append(v)

# ====== 그림 설정 ======
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(-0.1, width + 0.1)
ax.set_ylim(-0.1, height + 0.1)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

rects = []
centers = []
scatters = []
texts = []

for (x0, y0) in origins:
    # 박스
    rect = Rectangle((x0, y0), BOX_SIZE, BOX_SIZE, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    rects.append(rect)

    # 중심 고정점
    cx, cy = x0 + BOX_SIZE/2, y0 + BOX_SIZE/2
    (center_artist,) = ax.plot(cx, cy, marker='o', markersize=4, linestyle='None')
    centers.append(center_artist)

    # 이동 점 (초기 비움)
    scatter = ax.scatter([], [], s=16)
    scatters.append(scatter)

    # 사분면 카운트 표시
    t = ax.text(x0 + 0.03, y0 + BOX_SIZE - 0.08, "", fontsize=8, family="monospace")
    texts.append(t)

def init():
    for sc in scatters:
        sc.set_offsets(np.empty((0, 2)))
    for t in texts:
        t.set_text("")
    return (*scatters, *texts)

def update(frame):
    artists = []
    for i, (x0, y0) in enumerate(origins):
        pos = positions[i]
        vel = velocities[i]

        # 위치 업데이트
        pos += vel

        # 벽 반사 (로컬 좌표 [0, BOX_SIZE] 유지)
        hit_left = pos[:, 0] < 0.0
        hit_right = pos[:, 0] > BOX_SIZE
        hit_bottom = pos[:, 1] < 0.0
        hit_top = pos[:, 1] > BOX_SIZE

        vel[hit_left | hit_right, 0] *= -1
        vel[hit_bottom | hit_top, 1] *= -1

        pos[:, 0] = np.clip(pos[:, 0], 0.0, BOX_SIZE)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, BOX_SIZE)

        # 전역 좌표로 변환해서 산점도 갱신
        gx = x0 + pos[:, 0]
        gy = y0 + pos[:, 1]
        scatters[i].set_offsets(np.c_[gx, gy])
        artists.append(scatters[i])

        # 중심 기준 사분면 카운트
        cx, cy = x0 + BOX_SIZE/2, y0 + BOX_SIZE/2
        dx, dy = gx - cx, gy - cy
        q1 = np.sum((dx >= 0) & (dy >= 0))  # 우상
        q2 = np.sum((dx <  0) & (dy >= 0))  # 좌상
        q3 = np.sum((dx <  0) & (dy <  0))  # 좌하
        q4 = np.sum((dx >= 0) & (dy <  0))  # 우하
        texts[i].set_text(f"Q1:{q1:2d} Q2:{q2:2d}\nQ3:{q3:2d} Q4:{q4:2d}")
        artists.append(texts[i])

    return tuple(artists)

anim = animation.FuncAnimation(
    fig, update, init_func=init, frames=FRAMES, interval=INTERVAL, blit=True
)

# 1) 그냥 실행해서 창으로 보기
plt.show()

# 2) (선택) HTML로 저장해서 브라우저에서 보기
# html = anim.to_jshtml()
# with open("squares_animation.html", "w", encoding="utf-8") as f:
#     f.write(html)
# print("Saved: squares_animation.html")
