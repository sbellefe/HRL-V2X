import os, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import math
import pandas as pd
import numpy as np


class HighwayVisualizer:
    """
    A Pygame-based visualizer for a 1 km, 6-lane highway (3 lanes per direction),
    each lane 4 m wide => total 24 m. The road is drawn horizontally across the
    screen, with green visible at the top and bottom. Vehicles are scaled up
    so they appear more visible compared to the road length.

    Usage:
      - Initialize the class.
      - Call draw_frame(...) with a Pandas DataFrame of positions and a dictionary w/ pathloss info
      - The DataFrame should have columns:
          [time, id, x, y, angle, speed, lane]
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

    def draw_frame(self, pos_sample_df, position_info, t0, testing, fps=0):
        """
        Renders a frame from a DataFrame containing columns:
            ['time', 'id', 'x', 'y', 'angle', 'speed', 'lane']
        and a dictionary (position_info) with the following keys:
            'pl_v2i_v2v': interference channel between V2V and V2I vehicles [m, i*2]
            'pl_v2i_bs': channel between V2I vehicles and base station [m]
            'pl_v2v_bs': interference between V2V vehicles and base station [i*2]
            'pl_v2v_v2v': pathloss matrix between V2V vehicles [i*2, i*2]

        - pos_sample_df: DataFrame with one row per vehicle
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
        self._draw_scale_bar()

        # 3) Build a list of vehicle data from the DataFrame TODO change so that we accept Vehicle lists instead of pos data?
        vehicles = []
        v2i_count, v2v_count, idx = 1, 1, 0
        for _, row in pos_sample_df.iterrows():
            veh_type = 'V2I' if 'V2I' in row['id'] else 'V2V'
            if veh_type == 'V2I':
                label = f"V2I-{v2i_count}"
                v2i_count += 1
            else:
                label = f"V2V-{v2v_count}"
                v2v_count += 1

            vehicles.append({
                'id': row['id'],
                'x': row['x'],
                'y': row['y'],
                'angle': row['angle'],
                'speed': row['speed'],
                'lane': row['lane'],
                'idx': idx,
                'type': veh_type,
                'label': label,
            })
            idx += 1

        # 4) Draw vehicles
        for veh in vehicles:
            color = self.COLOR_RED if veh['type'] == 'V2I' else self.COLOR_BLUE
            self._draw_vehicle(veh['x'], veh['y'], veh['angle']-90, color, label=veh['label'])

        # 5) Optionally draw V2V arrows
        self._draw_v2v_arrows(vehicles)

        # 6) Draw the white textbox with pathloss information in the top green area
        self._draw_pathloss_textbox(position_info, vehicles)

        #bottom left text
        mode = "Testing" if testing else "Training"
        self._draw_text(f"{mode} Dataset Time index: {t0:.1f} seconds", self.screen_width/2, 480, font_size=20, align="center")

        # 7) Flip display
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

        # Center (solid yellow) line at y=0
        pygame.draw.line(self.screen, self.COLOR_YELLOW,
                         (self._mx_to_px(0), self._my_to_py(0)),
                         (self._mx_to_px(self.road_length_m), self._my_to_py(0)), 3)

        self._draw_base_station()

        # White dashed lines for lane boundaries
        lane_boundaries = [self.lane_width_m, 2*self.lane_width_m]
        for y_m in lane_boundaries:
            self._draw_dashed_line(self.COLOR_WHITE, 2, (0, y_m), (self.road_length_m, y_m))
            self._draw_dashed_line(self.COLOR_WHITE, 2, (0, -y_m), (self.road_length_m, -y_m))

    def _draw_vehicle(self, x_m, y_m, angle_deg, color, label=None):
        """
        Draw a vehicle as a scaled, rotated rectangle with a small arrow on top.
        If a label is provided, overlay it near the vehicle's center.

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

        # Overlay the vehicle label near its center (using a small font)
        if label:
            self._draw_text(label, rect.centerx, rect.centery-12, font_size=15, color=(0, 0, 0))

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
        Render multi-line text on the screen with the given alignment.
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


    def _draw_pathloss_textbox(self, position_info, vehicles):
        """
        Draws a white textbox in the top area of the screen and prints the
        pathloss information using the vehicle labels.
        """
        # # Define the rectangle for the textbox (top area above the road)
        # border = 5
        # textbox_rect = pygame.Rect(border, border, self.screen_width-2*border, self.road_offset_y-2*border)
        # pygame.draw.rect(self.screen, self.COLOR_WHITE, textbox_rect)

        # Formatting
        font_size = 14
        text_colour = (0,0,0)
        backgrnd_colour = (255,255,255)
        border_colour = (5,8,55)
        line_spacing = font_size + 2
        x = 10  # left margin
        y = 10   # top margin

        # #Add title
        # title_y = y  # Keep track of the initial y for the title
        # self._draw_text("Pathloss Information (dB)", textbox_rect.centerx, title_y, font_size=16, color=(0, 0, 0))
        # y += line_spacing   # Add a little extra space after the title


        # for key, value in position_info.items(): print(key, value.shape)


        """V2I-V2V"""
        row_titles = ["V2I-1", "V2I-2", "V2I-3", "V2I-4"]
        col_titles = ["V2V-1", "V2V-2", "V2V-3", "V2V-4", "V2V-5", "V2V-6", "V2V-7", "V2V-8"]
        table_v2i_v2v = Table(position_info['pl_v2i_v2v'], row_titles, col_titles,
                              "Pathloss [dB]: V2I Vehicles - V2V Vehicles",
                              font_size=font_size, text_color=text_colour, background_color=backgrnd_colour,
                              border_color=border_colour, cell_padding=5)
        table_position = (10, 15)  # example coordinates within your textbox
        table_v2i_v2v.draw(self.screen, table_position)

        table_d_v2i_v2v = Table(position_info['d_v2i_v2v'], row_titles, col_titles,
                              "Distance [m]: V2I Vehicles - V2V Vehicles",
                              font_size=font_size, text_color=text_colour, background_color=backgrnd_colour,
                              border_color=border_colour, cell_padding=5)
        table_position = (10, 460)  # example coordinates within your textbox
        table_d_v2i_v2v.draw(self.screen, table_position)


        """V2I-BS"""
        row_titles = ["BS"]
        col_titles = ["V2I-1", "V2I-2", "V2I-3", "V2I-4"]
        table_v2i_bs = Table(position_info['pl_v2i_bs'], row_titles, col_titles,
                              "Pathloss [dB]: V2I Vehicles - Base Station",
                              font_size=font_size, text_color=text_colour, background_color=backgrnd_colour,
                              border_color=border_colour, cell_padding=5)
        table_position = (470, 15)  # example coordinates within your textbox
        table_v2i_bs.draw(self.screen, table_position)

        """V2V-BS"""
        row_titles = ["BS"]
        col_titles = ["V2V-1", "V2V-2", "V2V-3", "V2V-4", "V2V-5", "V2V-6", "V2V-7", "V2V-8"]
        table_v2v_bs = Table(position_info['pl_v2v_bs'], row_titles, col_titles,
                             "Pathloss [dB]: V2V Vehicles - Base Station",
                             font_size=font_size, text_color=text_colour, background_color=backgrnd_colour,
                             border_color=border_colour, cell_padding=5)
        table_position = (400, 80)  # example coordinates within your textbox
        table_v2v_bs.draw(self.screen, table_position)

        """V2V-V2V"""
        row_titles = ["V2V-1", "V2V-2", "V2V-3", "V2V-4", "V2V-5", "V2V-6", "V2V-7", "V2V-8"]
        col_titles = ["V2V-1", "V2V-2", "V2V-3", "V2V-4", "V2V-5", "V2V-6", "V2V-7", "V2V-8"]
        table_v2v_v2v = Table(position_info['pl_v2v_v2v'], row_titles, col_titles,
                             "Pathloss [dB]: V2V Vehicles - V2V Vehicles",
                             font_size=font_size, text_color=text_colour, background_color=backgrnd_colour,
                             border_color=border_colour, cell_padding=5,
                             row_height=12)
        table_position = (800, 10)  # example coordinates within your textbox
        table_v2v_v2v.draw(self.screen, table_position)

        table_d_v2v_v2v = Table(position_info['d_v2v_v2v'], row_titles, col_titles,
                              "Distance [m]: V2V Vehicles - V2V Vehicles",
                              font_size=font_size, text_color=text_colour, background_color=backgrnd_colour,
                              border_color=border_colour, cell_padding=5,
                              row_height=12)
        table_position = (800, 460)  # example coordinates within your textbox
        table_d_v2v_v2v.draw(self.screen, table_position)

        # #[[row_titles], [column titles], main title,
        #
        #
        # table_data_example = [["V2V-1", "V2V-2", "V2V-3"...], ["V2I-1", "V2I-2", "V2I-3"...], "V2V Vehcles - V2I Vehicles",]
        # my_table1 = Table(table_data_example)
        # my_table1.draw(...)


        """OLD: New code should follow style above and use the new Table method"""
        # lines = []
        #
        #
        # # Separate vehicles by type
        # v2i = [veh for veh in vehicles if veh['type'] == 'V2I']
        # v2v = [veh for veh in vehicles if veh['type'] == 'V2V']
        #
        # # --- V2I-V2V ---
        # lines.append("V2I Vehicles - V2V Vehicles:")
        # # Expecting pl_v2i_v2v to be of shape [#V2I, #V2V]
        # pl_v2i_v2v = position_info.get('pl_v2i_v2v')
        # for i, veh in enumerate(v2i):
        #     line = f"  {veh['label']} -> "
        #     sub_entries = []
        #     for j, vveh in enumerate(v2v):
        #         try:
        #             val = pl_v2i_v2v[i, j]
        #         except (IndexError, TypeError):
        #             val = 0
        #         sub_entries.append(f"{vveh['label']}: {val:.1f}")
        #     line += "; ".join(sub_entries)
        #     lines.append(line)
        #
        # # --- V2I-eNB ---
        # lines.append("V2I Vehicles - Base Station:")
        # pl_v2i_eNB = position_info.get('pl_v2i_bs')
        # for i, veh in enumerate(v2i):
        #     try:
        #         val = pl_v2i_eNB[i]
        #     except (IndexError, TypeError):
        #         val = 0
        #     lines.append(f"  {veh['label']}: {val:.1f}")
        #
        # # --- V2V-eNB ---
        # lines.append("V2V Vehicles - Base Station:")
        # pl_v2v_eNB = position_info.get('pl_v2v_bs')
        # for j, veh in enumerate(v2v):
        #     try:
        #         val = pl_v2v_eNB[j]
        #     except (IndexError, TypeError):
        #         val = 0
        #     lines.append(f"  {veh['label']}: {val:.1f}")
        #
        # # --- V2V-V2V ---
        # lines.append("V2V Vehicles - V2V Vehicles:")
        # pl_v2v_v2v = position_info.get('pl_v2v_v2v')
        # # Create a header row: one column per V2V vehicle
        # header = "       "
        # for veh in v2v:
        #     header += f"{veh['label']}      "
        # lines.append(header)
        # # Each row corresponds to one V2V vehicle
        # num_rows = len(v2v)
        # for i in range(num_rows):
        #     row_label = f"{v2v[i]['label']}: "
        #     row_str = row_label.ljust(7)
        #     for j in range(num_rows):
        #         try:
        #             val = pl_v2v_v2v[i, j]
        #         except (IndexError, TypeError):
        #             val = 0
        #         row_str += f"{val:6.1f} "
        #     lines.append(row_str)
        #
        # # Render each line in the textbox
        # for line in lines:
        #     self._draw_text(line, x, y, font_size=font_size, color=(0, 0, 0), align="left")
        #     y += line_spacing

    def _draw_scale_bar(self):
        """
        Draw a scale bar with its bottom-left corner controlled by
        the variables `corner_x` and `corner_y`.

        Horizontal: 100 m long with tick marks every 20 m.
        Vertical: 5 m tall with tick marks every 1 m.
        All lines and text are drawn in black.
        """
        # Bottom-left corner of the scale bar (can be adjusted as needed)
        corner_x = 380
        corner_y = self.screen_height - 25

        thickness = 2
        scale_color = (0, 0, 0)  # black

        # ----- Horizontal Scale Bar -----
        x_scale_m = 100
        x_scale_px = int(x_scale_m * self.scale_x)
        # Draw the horizontal line from the corner to the right.
        start_h = (corner_x, corner_y)
        end_h = (corner_x + x_scale_px, corner_y)
        pygame.draw.line(self.screen, scale_color, start_h, end_h, thickness)

        # Draw tick marks every 20 m along the horizontal line.
        tick_interval_m = 20
        tick_interval_px = int(tick_interval_m * self.scale_x)
        tick_height = 5  # tick length in pixels
        for tick_px in range(0, x_scale_px + 1, tick_interval_px):
            tick_x = corner_x + tick_px
            # Draw vertical tick
            pygame.draw.line(self.screen, scale_color, (tick_x, corner_y), (tick_x, corner_y - tick_height), thickness)
            # Label the tick (showing distance in meters)
            tick_label = f"{int(tick_px / self.scale_x)}"
            self._draw_text(tick_label, tick_x, corner_y + 7, font_size=16, color=scale_color, align="center")

        # # Label the overall horizontal scale (centered below the line)
        mid_h_x = corner_x + x_scale_px // 2
        # self._draw_text(f"{x_scale_m} m", mid_h_x, corner_y + 20, font_size=16, color=scale_color, align="center")

        # ----- Vertical Scale Bar -----
        y_scale_m = 5
        y_scale_px = int(y_scale_m * self.scale_y)

        start_v = (corner_x, corner_y)
        end_v = (corner_x, corner_y - y_scale_px)
        pygame.draw.line(self.screen, scale_color, start_v, end_v, thickness)

        # Draw tick marks at 1m to 5m only
        tick_interval_m_vert = 1
        tick_width = 5  # horizontal tick length in pixels
        for m in range(1, y_scale_m + 1):  # Only show 1m to 5m
            tick_y = corner_y - int(m * self.scale_y)
            pygame.draw.line(self.screen, scale_color, (corner_x, tick_y), (corner_x + tick_width, tick_y), thickness)
            self._draw_text(f"{m}", corner_x - 5, tick_y-4, font_size=16, color=scale_color, align="right")

        # ----- Vertical Scale Bar -----
        # y_scale_m = 5
        # y_scale_px = int(y_scale_m * self.scale_y)
        # # Draw the vertical line upward from the corner.
        # start_v = (corner_x, corner_y)
        # end_v = (corner_x, corner_y - y_scale_px)
        # pygame.draw.line(self.screen, scale_color, start_v, end_v, thickness)
        #
        # # Draw tick marks every 1 m along the vertical line.
        # tick_interval_m_vert = 1
        # tick_interval_px_vert = int(tick_interval_m_vert * self.scale_y)
        # tick_width = 5  # tick length in pixels (horizontal extension)
        # for tick_px in range(0, y_scale_px + 1, tick_interval_px_vert):
        #     tick_y = corner_y - tick_px
        #     # Draw horizontal tick
        #     pygame.draw.line(self.screen, scale_color, (corner_x, tick_y), (corner_x + tick_width, tick_y), thickness)
        #     # Label the tick (to the left of the tick mark)
        #     tick_label = f"{int(tick_px / self.scale_y)}"
        #     self._draw_text(tick_label, corner_x - 5, tick_y, font_size=16, color=scale_color, align="right")

        # # Label the overall vertical scale (centered along the line)
        # mid_v_y = corner_y - y_scale_px // 2
        # self._draw_text(f"{y_scale_m} m", corner_x - 30, mid_v_y, font_size=16, color=scale_color, align="right")

        # ----Add title--
        self._draw_text(f"Scale [m]", mid_h_x+8, corner_y - 50, font_size=16, color=scale_color, align="center")

    def quit(self):
        """
        Gracefully quit Pygame.
        """
        pygame.quit()


class Table:
    def __init__(self, matrix, row_titles, col_titles, title, **kwargs):
        """
        Initialize the Table.

        :param matrix: 2D list of table cell values (or a 1D list, which will be wrapped).
        :param row_titles: List of row titles (displayed as the first column).
        :param col_titles: List of column titles (displayed as the first row).
        :param title: Overall table title drawn above the table.
        :param kwargs: Formatting options such as:
            - font_name (default None, uses pygame default)
            - font_size (default 14)
            - text_color (default (0, 0, 0))
            - background_color (default (255, 255, 255))
            - border_color (default (0, 0, 0))
            - cell_padding (default 5)
            - title_font_size (default same as font_size+2)
            - custom_row_height (optional: override computed row height for tighter margins)
        """
        # If the matrix is one-dimensional, wrap it into a 2D list.
        if isinstance(matrix, np.ndarray):
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)  # Make it a row vector
            elif matrix.ndim != 2:
                raise ValueError("Matrix must be 1D or 2D")
            matrix = matrix.tolist()
        elif isinstance(matrix, list):
            if matrix and not isinstance(matrix[0], list):
                matrix = [matrix]  # Wrap 1D list into a 2D list
        else:
            raise TypeError("Matrix must be a list or NumPy array")
        self.matrix = matrix
        self.row_titles = row_titles if row_titles is not None else []
        self.col_titles = col_titles if col_titles is not None else []
        self.title = title

        # Formatting options
        self.font_name = kwargs.get('font_name', None)
        self.font_size = kwargs.get('font_size', 14)
        self.text_color = kwargs.get('text_color', (0, 0, 0))
        self.background_color = kwargs.get('background_color', (255, 255, 255))
        self.border_color = kwargs.get('border_color', (0, 0, 0))
        self.cell_padding = kwargs.get('cell_padding', 5)
        self.title_font_size = kwargs.get('title_font_size', self.font_size + 2)

        # Create fonts
        self.font = pygame.font.SysFont(self.font_name, self.font_size)
        self.title_font = pygame.font.SysFont(self.font_name, self.title_font_size)

        # Precompute cell dimensions based on content and formatting.
        self._compute_dimensions()

        # Optional override for row height (e.g., to shrink the table for the V2V-V2V case)
        self.row_height = kwargs.get('row_height', self.row_height)

    def _compute_dimensions(self):
        """Compute the width of each column and the height of each row."""
        # Determine number of columns based on col_titles or matrix
        self.num_cols = len(self.col_titles) if self.col_titles else (len(self.matrix[0]) if self.matrix else 0)
        self.num_rows = len(self.row_titles) if self.row_titles else len(self.matrix)

        self.col_widths = []

        # First, calculate width for row title column if needed.
        if self.row_titles:
            max_width = 0
            for title in self.row_titles:
                text_surface = self.font.render(str(title), True, self.text_color)
                max_width = max(max_width, text_surface.get_width())
            # Also consider header cell for row titles if col_titles exist (could be empty)
            header_surface = self.font.render("", True, self.text_color)
            max_width = max(max_width, header_surface.get_width())
            self.col_widths.append(max_width + 2 * self.cell_padding)

        # Then, compute fixed widths for each table column using the column titles.
        for col in range(self.num_cols):
            if self.col_titles:
                header = self.col_titles[col]
                header_surface = self.font.render(str(header), True, self.text_color)
                col_width = header_surface.get_width() + 2 * self.cell_padding
            else:
                # Fallback: determine width from matrix content if no header is provided.
                max_width = 0
                for row in range(len(self.matrix)):
                    try:
                        cell = self.matrix[row][col]
                    except IndexError:
                        cell = ""
                    cell_surface = self.font.render(str(cell), True, self.text_color)
                    max_width = max(max_width, cell_surface.get_width())
                col_width = max_width + 2 * self.cell_padding
            self.col_widths.append(col_width)

        # Uniform row height (based on font height plus padding)
        self.row_height = self.font.get_height() + 2 * self.cell_padding
        # Header row height (if column titles exist)
        self.header_height = self.row_height if self.col_titles else 0
        # Overall title height (if title exists)
        self.title_height = self.title_font.get_height() + self.cell_padding if self.title else 0

    def draw(self, surface, topleft):
        """
        Draw the table onto the given pygame surface starting at topleft.

        :param surface: The pygame surface on which to draw the table.
        :param topleft: (x, y) coordinate tuple for the top-left corner of the table.
        """
        x0, y0 = topleft

        # Draw overall title if present.
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.text_color)
            title_rect = title_surface.get_rect(centerx=x0 + sum(self.col_widths) // 2, y=y0)
            surface.blit(title_surface, title_rect)
            y0 += self.title_height  # shift drawing down after the title

        # Draw header row if column titles exist.
        if self.col_titles:
            header_y = y0
            x = x0
            # If there is a row title column, draw an empty header cell.
            if self.row_titles:
                cell_rect = pygame.Rect(x, header_y, self.col_widths[0], self.header_height)
                pygame.draw.rect(surface, self.background_color, cell_rect)
                pygame.draw.rect(surface, self.border_color, cell_rect, 1)
                x += self.col_widths[0]
            # Draw each header cell.
            for i, header in enumerate(self.col_titles):
                cell_width = self.col_widths[i + (1 if self.row_titles else 0)]
                cell_rect = pygame.Rect(x, header_y, cell_width, self.header_height)
                pygame.draw.rect(surface, self.background_color, cell_rect)
                pygame.draw.rect(surface, self.border_color, cell_rect, 1)
                header_surface = self.font.render(str(header), True, self.text_color)
                header_rect = header_surface.get_rect(center=cell_rect.center)
                surface.blit(header_surface, header_rect)
                x += cell_width
            y0 += self.header_height

        # Draw table rows.
        for row in range(len(self.matrix)):
            x = x0
            row_y = y0 + row * self.row_height
            # Draw row title if present.
            if self.row_titles:
                cell_rect = pygame.Rect(x, row_y, self.col_widths[0], self.row_height)
                pygame.draw.rect(surface, self.background_color, cell_rect)
                pygame.draw.rect(surface, self.border_color, cell_rect, 1)
                row_title = self.row_titles[row] if row < len(self.row_titles) else ""
                title_surface = self.font.render(str(row_title), True, self.text_color)
                title_rect = title_surface.get_rect(center=cell_rect.center)
                surface.blit(title_surface, title_rect)
                x += self.col_widths[0]
            # Draw each cell in the row with one floating-point value.
            for col in range(self.num_cols):
                cell_rect = pygame.Rect(x, row_y, self.col_widths[col + (1 if self.row_titles else 0)], self.row_height)
                pygame.draw.rect(surface, self.background_color, cell_rect)
                pygame.draw.rect(surface, self.border_color, cell_rect, 1)
                try:
                    cell_value = self.matrix[row][col]
                except IndexError:
                    cell_value = ""
                # Format the cell value as a float with one decimal if possible.
                try:
                    text = f"{float(cell_value):.1f}"
                except (ValueError, TypeError):
                    text = str(cell_value)
                cell_surface = self.font.render(text, True, self.text_color)
                cell_text_rect = cell_surface.get_rect(center=cell_rect.center)
                surface.blit(cell_surface, cell_text_rect)
                x += self.col_widths[col + (1 if self.row_titles else 0)]