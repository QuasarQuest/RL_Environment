// src/world/grid.rs

use super::tile::Tile;

#[derive(Clone)]
pub struct Grid {
    tiles:      Vec<Tile>,
    pub width:  usize,
    pub height: usize,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            tiles:  vec![Tile::Free; width * height],
            width,
            height,
        }
    }

    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    pub fn get(&self, x: i32, y: i32) -> Option<Tile> {
        if self.in_bounds(x, y) {
            Some(self.tiles[y as usize * self.width + x as usize])
        } else {
            None
        }
    }

    pub fn set(&mut self, x: i32, y: i32, tile: Tile) {
        if self.in_bounds(x, y) {
            self.tiles[y as usize * self.width + x as usize] = tile;
        }
    }

    pub fn is_walkable(&self, x: i32, y: i32) -> bool {
        self.get(x, y).map(|t| t.is_walkable()).unwrap_or(false)
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, Tile)> + '_ {
        (0..self.height).flat_map(move |y| {
            (0..self.width).map(move |x| (x, y, self.tiles[y * self.width + x]))
        })
    }

    /// Direct tile read without bounds check — caller must guarantee in-bounds.
    #[inline]
    pub(crate) unsafe fn get_unchecked(&self, x: i32, y: i32) -> Tile {
        *self.tiles.get_unchecked(y as usize * self.width + x as usize)
    }
}
