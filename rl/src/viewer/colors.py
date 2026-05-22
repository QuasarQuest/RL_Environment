# viewer/colors.py — color palette
from pyray import Color

# Tiles
FREE       = Color(22,  22,  26,  255)
OBSTACLE   = Color(58,  58,  64,  255)
BASE       = [Color(30,  90, 210, 255), Color(200, 55,  55,  255)]
SAFEZONE   = [Color(30,  90, 210,  35), Color(200, 55,  55,  35)]
GRID_LINE  = Color(38,  38,  44,  255)

# Agents  (fill, outline)
AGENT_FILL    = [Color(70, 150, 255, 255), Color(255,  85,  85, 255)]
AGENT_OUTLINE = [Color(20,  70, 190, 255), Color(190,  25,  25, 255)]

# Items
ITEM = [
    Color(255, 200,  30, 255),   # 0 Gold
    Color(240,  70,  70, 255),   # 1 Health
    Color( 70, 160, 255, 255),   # 2 Ammo
    Color( 70, 220, 120, 255),   # 3 SpeedBoost
]

# HUD
HUD_BG     = Color(14,  14,  18, 230)
HUD_TEXT   = Color(210, 210, 215, 255)
HUD_DIM    = Color(110, 110, 120, 255)
HUD_BAR    = Color( 80, 160, 255, 255)
HUD_SCORE  = Color(255, 200,  50, 255)

# End screen
END_OVERLAY = Color(0, 0, 0, 175)
END_SCORE   = Color(255, 200, 50, 255)
