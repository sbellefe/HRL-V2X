import os, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import math
import pandas as pd


class HighwayVisualizer:
    """
    A Pygame-based visualizer for a 1 km, 6-lane highway (3 lanes per direction),
    each lane 4 m wide => total 24 m. The road is drawn horizontally across the
    screen, with green visible at the top and bottom. Vehicles are scaled up
    so they appear more visible compared to the road length.

    Usage:
      - Initialize the class (this won't open a window immediately).
      - Call draw_frame(...) with a Pandas DataFrame of positions to open/update
        the window.
      - The DataFrame must have columns:
          [time, id, x, y, angle, speed, lane]
        where x in [0..1000], y in [-12..+12], angle in degrees (0=right, 90=up, etc.).
      - The window remains open and responsive until you call quit() or close it.
    """

    def __init__(self):
        """
        We won't create the Pygame window until the first draw_frame() call.
        """
        pygame.init()
        self.screen = None
        pygame.display.set_caption("V2X Highway Visualization")

        # Screen / window size
        self.screen_width = 1200
        self.screen_height = 600

        # Road geometry in meters
        self.road_length_m = 1000 # 1km

        lane_count = 6
        self.lane_width_m = 3.2
        self.road_width_m = self.lane_width_m * lane_count
        self.eNB_xy = [500, -43]  # position of base station

        # We'll draw the road to occupy 50% of the window's vertical dimension
        self.road_height_px = int(0.5 * self.screen_height)
        self.road_offset_y = (self.screen_height - self.road_height_px) // 2

        # Horizontal scale: entire 1000 m => entire screen_width
        self.scale_x = self.screen_width / float(self.road_length_m)
        # Vertical scale: 24 m => self.road_height_px
        self.scale_y = self.road_height_px / float(self.road_width_m)

        # Colors
        self.COLOR_GRASS = (0, 150, 0)
        self.COLOR_ROAD = (100, 100, 100)
        self.COLOR_YELLOW = (255, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLUE = (0, 0, 255)
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_ARROW = (0, 200, 0)  # For V2V arrow lines

        # Vehicle real-world dimensions
        self.vehicle_width_m = 2.5
        self.vehicle_length_m = 5.0

        # Make vehicles bigger relative to the road
        self.VEHICLE_SCALE_FACTOR = 3.5

        # Pygame clock
        self.clock = pygame.time.Clock()

    def draw_frame(self, pos_sample_df, v2v_pairs=None, fps=0):
        """
        Renders a frame from a DataFrame containing columns:
            ['time', 'id', 'x', 'y', 'angle', 'speed', 'lane']

        - pos_sample_df: DataFrame with one row per vehicle
        - v2v_pairs: optional list of (i, j) pairs for drawing arrows
        - fps: if > 0, limit the frame rate

        The Pygame window is created on the first call.
        """
        # Create the window if not already
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # Process events to avoid freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
                return

        if fps > 0:
            self.clock.tick(fps)

        # 1) Fill the background with green (grass)
        self.screen.fill(self.COLOR_GRASS)

        # 2) Draw the road rectangle
        self._draw_road()

        # 3) Build a list of vehicle data from the DataFrame
        vehicles, idx = [], 0
        for _, row in pos_sample_df.iterrows():
            vehicles.append({
                'id': row['id'],
                'x': row['x'],
                'y': row['y'],
                'angle': row['angle'],
                'speed': row['speed'],
                'lane': row['lane'],
                'type': 'V2I' if 'V2I' in row['id'] else 'V2V',
                'idx': idx,
            })
            idx += 1

        # 4) Draw vehicles
        for veh in vehicles:
            color = self.COLOR_RED if veh['type'] == 'V2I' else self.COLOR_BLUE
            self._draw_vehicle(veh['x'], veh['y'], veh['angle']-90, color)

        # 5) Optionally draw V2V arrows
        self._draw_v2v_arrows(vehicles)

        # 6) Flip display
        pygame.display.flip()

    def _draw_road(self):
        """
        Draw the central road rectangle, plus dashed lane lines:
          - The road extends horizontally across the entire screen (0..screen_width).
          - Vertically from road_offset_y..road_offset_y+road_height_px.
          - A yellow dashed line at y=0 (center).
          - White dashed lines at y=±4, ±8, ±12 for lane boundaries (if 6 lanes).
        """
        # Road rectangle
        pygame.draw.rect(
            self.screen,
            self.COLOR_ROAD,
            (0, self.road_offset_y, self.screen_width, self.road_height_px)
        )

        # Lane lines: For a 6-lane highway (3 each direction),
        # y=0 is center, ±4 is boundary between lane 1 & 2, ±8 between lane 2 & 3, ±12 is outer boundary.
        # We'll draw them as dashed lines horizontally from x=0..road_length_m.

        # The center line: y=0 in road coords => dashed yellow
        # The center line: y=0 in road coords => solid yellow
        pygame.draw.line(self.screen, self.COLOR_YELLOW,
                         (self._mx_to_px(0), self._my_to_py(0)),
                         (self._mx_to_px(self.road_length_m), self._my_to_py(0)), 3
        )

        self._draw_base_station()

        # White dashed lines for lane boundaries
        lane_boundaries = [self.lane_width_m, 2*self.lane_width_m]
        for y_m in lane_boundaries:
            # +y
            self._draw_dashed_line(self.COLOR_WHITE, 2, (0, y_m), (self.road_length_m, y_m))
            # -y
            self._draw_dashed_line(self.COLOR_WHITE, 2, (0, -y_m), (self.road_length_m, -y_m))

    def _draw_vehicle(self, x_m, y_m, angle_deg, color):
        """
        Draw a vehicle as a scaled, rotated rectangle with a small arrow on top.

        x_m, y_m in [meters].
        angle_deg: 0=right, 90=up, 180=left, 270=down, etc.
        color: vehicle fill color (RGB).
        """
        # Convert real-world dims to pixels, scale up further to make vehicles bigger
        width_px = self.vehicle_width_m * self.scale_x * self.VEHICLE_SCALE_FACTOR
        length_px = self.vehicle_length_m * self.scale_x * self.VEHICLE_SCALE_FACTOR

        # Create a surface for the vehicle
        vehicle_surf = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
        vehicle_surf.fill(color)

        # Draw a small arrow shape on top
        arrow_color = self.COLOR_WHITE
        arrow_points = [
            (length_px * 0.8, width_px / 2),
            (length_px * 0.6, width_px * 0.3),
            (length_px * 0.6, width_px * 0.7)
        ]
        pygame.draw.polygon(vehicle_surf, arrow_color, arrow_points)

        # Rotate
        rotated_surf = pygame.transform.rotate(vehicle_surf, -angle_deg)

        # Center position in screen coords
        cx = self._mx_to_px(x_m)
        cy = self._my_to_py(y_m)
        rect = rotated_surf.get_rect(center=(cx, cy))

        self.screen.blit(rotated_surf, rect)

    def _draw_base_station(self):
        """
        Draws a simple triangular base station at self.eNB_xy with a label.
        """
        # Convert coordinates
        x = self._mx_to_px(self.eNB_xy[0])
        y = self._my_to_py(self.eNB_xy[1] * 0.4)  # Adjust scaling

        # Define tower shape
        tower_height = 30
        tower_width = 15
        tower_points = [(x, y - tower_height), (x - tower_width, y + tower_height), (x + tower_width, y + tower_height)]

        # Draw tower and red light
        pygame.draw.polygon(self.screen, (100, 100, 100), tower_points)
        pygame.draw.circle(self.screen, (255, 0, 0), (x, y - tower_height), 7)

        # Add label using `_draw_text`
        self._draw_text("Base\nStation", x+40, y - tower_height - 10, font_size=20)


    def _draw_text(self, text, x, y, font_size=16, color=(255, 255, 255), align="center"):
        """
        General method to render text on the screen.
        Supports multi-line text and alignment.
        """
        font = pygame.font.Font(None, font_size)
        lines = text.split("\n")  # Handle multi-line text
        line_height = font_size + 2  # Spacing between lines

        for i, line in enumerate(lines):
            text_surface = font.render(line, True, color)
            text_rect = text_surface.get_rect()

            if align == "center":
                text_rect.center = (x, y + i * line_height)
            elif align == "left":
                text_rect.topleft = (x, y + i * line_height)
            elif align == "right":
                text_rect.topright = (x, y + i * line_height)

            self.screen.blit(text_surface, text_rect)

    def _draw_v2v_arrows(self, vehicles):
        """
        Draw simple lines between vehicles that have V2V links.
        v2v_pairs: list of (i, j) where i,j are indices in 'vehicles' list.
        """


        v2v_pairs = []
        for veh in vehicles:
            if veh['type'] == 'V2V' and veh['id'].endswith('0.0'):
                v2v_pairs.append((veh['idx'], veh['idx']+1))


        for (i, j) in v2v_pairs:
            if i < len(vehicles) and j < len(vehicles):
                x1, y1 = vehicles[i]['x'], vehicles[i]['y']
                x2, y2 = vehicles[j]['x'], vehicles[j]['y']
                start = (self._mx_to_px(x1), self._my_to_py(y1))
                end = (self._mx_to_px(x2), self._my_to_py(y2))
                pygame.draw.line(self.screen, self.COLOR_BLUE, start, end, 2)  # Thin blue line

    def _draw_arrow_line(self, start, end, color, width=2, arrow_size=10):
        """
        Draw a line from start->end with an arrowhead at the end.
        """
        pygame.draw.line(self.screen, color, start, end, width)
        dx, dy = end[0] - start[0], end[1] - start[1]
        angle = math.atan2(dy, dx)
        # Arrowhead
        arrow_x = end[0] - arrow_size * math.cos(angle)
        arrow_y = end[1] - arrow_size * math.sin(angle)
        left = (
            arrow_x + (arrow_size / 2.0) * math.sin(angle),
            arrow_y - (arrow_size / 2.0) * math.cos(angle)
        )
        right = (
            arrow_x - (arrow_size / 2.0) * math.sin(angle),
            arrow_y + (arrow_size / 2.0) * math.cos(angle)
        )
        pygame.draw.polygon(self.screen, color, [end, left, right])

    def _draw_dashed_line(self, color, width, start_m, end_m, dash_length=20):
        """
        Draw a dashed line in 'road coordinates' from start_m->end_m.
        Each dash is dash_length in *pixels*.
        """
        start_px = (self._mx_to_px(start_m[0]), self._my_to_py(start_m[1]))
        end_px = (self._mx_to_px(end_m[0]), self._my_to_py(end_m[1]))
        x1, y1 = start_px
        x2, y2 = end_px
        total_length = math.hypot(x2 - x1, y2 - y1)
        dash_gap = dash_length * 2
        dashes = int(total_length // dash_gap)

        for i in range(dashes):
            start_frac = (i * dash_gap) / total_length
            end_frac = (i * dash_gap + dash_length) / total_length
            sx = x1 + (x2 - x1) * start_frac
            sy = y1 + (y2 - y1) * start_frac
            ex = x1 + (x2 - x1) * end_frac
            ey = y1 + (y2 - y1) * end_frac
            pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), width)

    def _mx_to_px(self, x_m):
        """
        Convert horizontal meters to screen pixels (0..road_length_m => 0..screen_width).
        """
        return x_m * self.scale_x

    def _my_to_py(self, y_m):
        """
        Convert vertical meters (y=0 is road center, ±12 for top/bottom of road)
        into screen coords. The road is drawn from self.road_offset_y to
        self.road_offset_y + self.road_height_px.

        y=0 => middle of the road in screen coords
        y= +12 => top boundary
        y= -12 => bottom boundary
        """
        # The center of the road in screen coords is road_offset_y + road_height_px/2
        center_screen_y = self.road_offset_y + self.road_height_px / 2
        return center_screen_y - (y_m * self.scale_y)

    def quit(self):
        """
        Gracefully quit Pygame.
        """
        pygame.quit()
