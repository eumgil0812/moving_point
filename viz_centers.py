import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D # For drawing lines between points and centers

# --- C++ Module (sim) Import ---
import sim

# ===== Parameters =====
BOX_WIDTH = 1.5  # Rectangular width
BOX_HEIGHT = 1.0 # Rectangular height
N = 10           # Number of moving points
SPEED = 0.02
FRAMES = 300
INTERVAL = 30
DT = 1.0
RADIUS = 0.5   # 커버리지 원 반경 (0.5로 유지)
K_NEIGHBORS = 3 # K-Nearest Neighbors to visualize

# [UPDATED] Centers in a 3x2 grid, K=6
# X-coords (3개), Y-coords (2개)를 사용하여 1.5 x 1.0 영역에 배치합니다.
x_steps = np.linspace(BOX_WIDTH / 6, BOX_WIDTH - BOX_WIDTH / 6, 3) # 3개: 0.25, 0.75, 1.25
y_steps = np.linspace(BOX_HEIGHT / 4, BOX_HEIGHT - BOX_HEIGHT / 4, 2) # 2개: 0.25, 0.75
xx, yy = np.meshgrid(x_steps, y_steps)
centers = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
K = centers.shape[0]  # K is now 6

# ===== State: Initialize points in a uniform grid within the rectangle =====
GRID_W = int(np.ceil(np.sqrt(N * BOX_WIDTH / BOX_HEIGHT)))
GRID_H = int(np.ceil(N / GRID_W))

xs = np.linspace(0.02, BOX_WIDTH - 0.02, GRID_W) # Use BOX_WIDTH
ys = np.linspace(0.02, BOX_HEIGHT - 0.02, GRID_H) # Use BOX_HEIGHT
xx, yy = np.meshgrid(xs, ys)
pts = np.c_[xx.ravel(), yy.ravel()][:N]
pos = pts.reshape(1, N, 2).astype(np.float64)

rng = np.random.default_rng(7)
theta = rng.uniform(0, 2*np.pi, size=(1, N))
vel = np.stack([np.cos(theta), np.sin(theta)], axis=2).astype(np.float64) * SPEED

# ===== Plot Setup =====
fig, ax = plt.subplots(figsize=(9,6)) # Adjust figure size for 1.5 aspect ratio
ax.set_xlim(0, BOX_WIDTH); ax.set_ylim(0, BOX_HEIGHT) # Adjust axis limits
ax.set_aspect('equal', adjustable='box')
ax.add_patch(Rectangle((0,0), BOX_WIDTH, BOX_HEIGHT, fill=False, linewidth=1.4)) # Draw rectangular boundary
ax.axis('off')
ax.set_title(f"K-Nearest Centers (N={N}, K={K}, K-NN={K_NEIGHBORS}) [Rectangular]", fontsize=10)

# Centers (black circles) + Coverage circles
ax.plot(centers[:,0], centers[:,1], 'o', markersize=8, color='black', zorder=5)
circles = []
for k in range(K):
    c = Circle((centers[k,0], centers[k,1]), RADIUS, edgecolor='gray',
               facecolor='none', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)
    ax.add_patch(c); circles.append(c)

# Point scatter plot (colored by nearest center)
cmap = plt.get_cmap('tab10')
colors_map = np.array([cmap(i % 10) for i in range(K)])
sc = ax.scatter([], [], s=12, c=[], zorder=2)

# K-Nearest Lines: Line2D objects to connect each point to its K_NEIGHBORS centers
knn_lines = []
for _ in range(N):
    # Create K_NEIGHBORS lines for each point
    lines_for_point = []
    for k in range(K_NEIGHBORS):
        # Line properties for thickness/alpha based on distance rank (0=closest, 2=3rd closest)
        line = Line2D([], [], color='gray', alpha=0.5 / (k + 1), linewidth=2.0 / (k + 1), zorder=3-k)
        ax.add_line(line)
        lines_for_point.append(line)
    knn_lines.append(lines_for_point)

# Text labels for each center (Voronoi count / Radius count)
texts = [ax.text(centers[k,0]+0.02, centers[k,1]+0.02, "",
                 fontsize=9, family="monospace",
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
                 zorder=10) for k in range(K)]

def init():
    # Initial state display
    current = pos.reshape(N, 2)
    sc.set_offsets(current)
    sc.set_color('gray')
    for t in texts: t.set_text("")
    
    # Hide lines initially
    for point_lines in knn_lines:
        for line in point_lines:
            line.set_data([], [])
            
    return (sc, *circles, *texts, *[line for lines in knn_lines for line in lines])


def update(_frame):
    # 1) Position Update (C++: Reflective boundary conditions for rectangular box)
    # Pass BOX_WIDTH and BOX_HEIGHT to the C++ function
    sim.update_boxes(pos, vel, BOX_WIDTH, BOX_HEIGHT, DT)

    # (Safety clamp for floating point errors)
    pos[0,:,0] = np.clip(pos[0,:,0], 0.0, BOX_WIDTH) 
    pos[0,:,1] = np.clip(pos[0,:,1], 0.0, BOX_HEIGHT) 

    # 2) Nearest Center Assignment & Radius Count
    current = pos.reshape(N,2)
    
    # Nearest 1 center assignment (for coloring)
    assign, counts = sim.nearest_assign(current, centers, BOX_WIDTH)
    
    # K_NEIGHBORS (3) nearest centers index calculation (for line visualization)
    k_assign_indices = sim.k_nearest_assign(current, centers, k_val=K_NEIGHBORS)
    
    within = sim.count_within_radius_multi(current, centers, RADIUS)

    # 3) Update points and colors
    sc.set_offsets(current)
    # Note: K is now 6. colors_map size is 6. This is fine.
    sc.set_color(colors_map[np.asarray(assign, dtype=int)])

    # 4) Update K-NN lines (N * K_NEIGHBORS lines)
    for i in range(N): # For each point (i)
        px, py = current[i] # Point coordinates
        
        for k_idx in range(K_NEIGHBORS): # For the K_NEIGHBORS nearest centers
            center_idx = k_assign_indices[i, k_idx]
            cx, cy = centers[center_idx] # Center coordinates
            
            line = knn_lines[i][k_idx]
            line.set_data([px, cx], [py, cy])
            
            # Line color based on the nearest center's color (Voronoi assignment)
            color = colors_map[assign[i]]
            # Adjust alpha and linewidth based on distance rank
            line.set_color(color)
            line.set_alpha(0.6 - (k_idx * 0.2)) # 1st: 0.6, 2nd: 0.4, 3rd: 0.2
            line.set_linewidth(1.8 - (k_idx * 0.5))
            

    # 5) Update text labels
    # Note: Text labels must be created for all K=6 centers
    for k in range(K):
        # Ensure we don't access counts/within beyond its bounds if N < K
        v_count = int(counts[k]) if k < len(counts) else 0
        r_count = int(within[k]) if k < len(within) else 0
        texts[k].set_text(f"V:{v_count:<3} R:{r_count:<3}")

    return (sc, *circles, *texts, *[line for lines in knn_lines for line in lines])

# Set blit=False for better stability in Matplotlib HTML rendering environments
anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=FRAMES, interval=INTERVAL, blit=False)

try:
    plt.show()
except Exception:
    pass

with open("centers_animation.html", "w", encoding="utf-8") as f:
    f.write(anim.to_jshtml())
print("Saved: centers_animation.html")
