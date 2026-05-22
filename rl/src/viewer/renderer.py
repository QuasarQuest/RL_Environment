# viewer/renderer.py
from __future__ import annotations
import pyray as rl
from . import colors as C

HUD_H = 64   # pixels reserved for the HUD strip at the bottom


class Renderer:
    def __init__(self, grid_w: int, grid_h: int, target_px: int = 720):
        self.gw = grid_w
        self.gh = grid_h
        self.ts = max(6, target_px // max(grid_w, grid_h))  # tile size in px
        self.win_w = grid_w * self.ts
        self.win_h = grid_h * self.ts + HUD_H

    # ── coordinate helpers ────────────────────────────────────────────────────

    def sx(self, gx: int) -> int:
        """Grid x → screen x (left edge of tile)."""
        return gx * self.ts

    def sy(self, gy: int) -> int:
        """Grid y → screen y. Flips y so y=0 is at the visual bottom."""
        return (self.gh - 1 - gy) * self.ts

    # ── public draw calls ────────────────────────────────────────────────────

    def draw_world(
        self,
        tiles:       list[int],
        agents:      list[tuple[int, int, int, int, int]],
        items:       list[tuple[int, int, int]],
        tick:        int,
        match_ticks: int,
        paused:      bool,
        speed_label: str,
    ) -> None:
        ts = self.ts

        # ── tiles ─────────────────────────────────────────────────────────────
        for idx, tile in enumerate(tiles):
            gx = idx % self.gw
            gy = idx // self.gw
            x, y = self.sx(gx), self.sy(gy)

            if tile == 0:
                rl.draw_rectangle(x, y, ts, ts, C.FREE)
            elif tile == 1:
                rl.draw_rectangle(x, y, ts, ts, C.OBSTACLE)
            elif tile >= 20:                       # SafeZone
                team = tile - 20
                rl.draw_rectangle(x, y, ts, ts, C.FREE)
                rl.draw_rectangle(x, y, ts, ts, C.SAFEZONE[team % 2])
            elif tile >= 10:                       # Base
                team = tile - 10
                rl.draw_rectangle(x, y, ts, ts, C.BASE[team % 2])

        # ── grid lines (only when tiles are large enough to see them) ─────────
        if ts >= 10:
            for gx in range(self.gw + 1):
                rl.draw_rectangle(gx * ts, 0, 1, self.gh * ts, C.GRID_LINE)
            for gy in range(self.gh + 1):
                rl.draw_rectangle(0, gy * ts, self.gw * ts, 1, C.GRID_LINE)

        # ── items ─────────────────────────────────────────────────────────────
        item_r = max(2, ts // 5)
        for ix, iy, kind in items:
            cx = self.sx(ix) + ts // 2
            cy = self.sy(iy) + ts // 2
            rl.draw_circle(cx, cy, item_r, C.ITEM[kind % 4])

        # ── agents ────────────────────────────────────────────────────────────
        agent_r = max(3, ts // 2 - 2)
        for ax, ay, team, gold, _score in agents:
            cx = self.sx(ax) + ts // 2
            cy = self.sy(ay) + ts // 2
            t  = team % 2
            rl.draw_circle(cx, cy, agent_r + 2, C.AGENT_OUTLINE[t])
            rl.draw_circle(cx, cy, agent_r,     C.AGENT_FILL[t])
            # Small gold pip when carrying
            if gold > 0 and ts >= 14:
                pip_r = max(2, ts // 8)
                rl.draw_circle(cx + agent_r - pip_r, cy - agent_r + pip_r,
                               pip_r, C.ITEM[0])

        # ── HUD ───────────────────────────────────────────────────────────────
        hud_y = self.gh * ts
        rl.draw_rectangle(0, hud_y, self.win_w, HUD_H, C.HUD_BG)

        # progress bar
        bar_w = int(self.win_w * min(tick / max(match_ticks, 1), 1.0))
        rl.draw_rectangle(0, hud_y, bar_w, 3, C.HUD_BAR)

        score = sum(s for *_, s in agents)
        rl.draw_text(f"Tick  {tick} / {match_ticks}", 12, hud_y + 10, 18, C.HUD_TEXT)
        rl.draw_text(f"Score {score}",                12, hud_y + 36, 18, C.HUD_SCORE)

        status = f"{'PAUSED  ' if paused else ''}{speed_label}"
        tw = rl.measure_text(status, 18)
        rl.draw_text(status, self.win_w - tw - 12, hud_y + 23, 18, C.HUD_DIM)

        # controls hint (bottom-right, tiny)
        hint = "Space pause   < > speed   R restart   Q quit"
        hw = rl.measure_text(hint, 12)
        rl.draw_text(hint, self.win_w - hw - 8, hud_y + HUD_H - 16, 12, C.HUD_DIM)

    def draw_end_screen(self, agents: list) -> None:
        score = sum(s for *_, s in agents)
        cx    = self.win_w // 2
        cy    = (self.gh * self.ts) // 2

        rl.draw_rectangle(0, 0, self.win_w, self.win_h, C.END_OVERLAY)

        title = "Episode Done"
        rl.draw_text(title, cx - rl.measure_text(title, 40) // 2, cy - 60, 40, rl.WHITE)

        sc = f"Score: {score}"
        rl.draw_text(sc, cx - rl.measure_text(sc, 30) // 2, cy, 30, C.END_SCORE)

        hint = "R  restart        Q  quit"
        rl.draw_text(hint, cx - rl.measure_text(hint, 20) // 2, cy + 55, 20, rl.GRAY)
