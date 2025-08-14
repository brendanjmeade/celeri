#!/usr/bin/env python3
"""Triangulation Selection Schematic Generator

This script generates a vector graphic illustrating the auto-triangulation selection
strategy for triangular dislocation elements (TDE). The figure shows how different
regions are assigned different triangulation types to minimize computational artifacts
from interior edge singularities.

Output: triangulation_schematic.svg
"""

import io
import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


def make_svg_deterministic(svg_content):
    """Make SVG content deterministic by replacing random IDs with sequential ones."""
    # Find all unique IDs in the SVG
    id_pattern = r'id="([^"]+)"'
    url_pattern = r"url\(#([^)]+)\)"
    xlink_pattern = r'xlink:href="#([^"]+)"'

    # Collect all unique IDs
    ids = set()
    for match in re.finditer(id_pattern, svg_content):
        ids.add(match.group(1))
    for match in re.finditer(url_pattern, svg_content):
        ids.add(match.group(1))
    for match in re.finditer(xlink_pattern, svg_content):
        ids.add(match.group(1))

    # Create deterministic mapping
    id_mapping = {}
    for i, old_id in enumerate(sorted(ids)):
        id_mapping[old_id] = f"id_{i:03d}"

    # Replace all occurrences
    result = svg_content
    for old_id, new_id in id_mapping.items():
        result = result.replace(f'id="{old_id}"', f'id="{new_id}"')
        result = result.replace(f"url(#{old_id})", f"url(#{new_id})")
        result = result.replace(f'xlink:href="#{old_id}"', f'xlink:href="#{new_id}"')

    return result


def create_triangulation_schematic():
    """Create the triangulation selection schematic figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {
        "/": "#8598e1",  # Light blue
        "\\": "#ee6666",  # Light red
        "V": "#88dd88",  # Light green
    }

    # Blue background rectangle covering the full range
    background_rect = Rectangle((-0.7, -0.7), 1.4, 1.4, facecolor=colors["/"], alpha=1)
    ax.add_patch(background_rect)

    # Checkerboard pattern for backslash regions - 2 rectangles touching at origin
    # Top-right quadrant: \ region (r > 0, s > 0)
    quad_tr = Rectangle((0, 0), 0.6, 0.6, facecolor=colors["\\"], alpha=1)
    ax.add_patch(quad_tr)

    # Bottom-left quadrant: \ region (r < 0, s < 0)
    quad_bl = Rectangle((-0.6, -0.6), 0.6, 0.6, facecolor=colors["\\"], alpha=1)
    ax.add_patch(quad_bl)

    # Central region (circular for V triangulation)
    central_radius = 0.1
    central_circle = Circle(
        (0, 0),
        central_radius,
        facecolor=colors["V"],
        alpha=1,
        edgecolor=colors["V"],
        linewidth=0,
    )
    ax.add_patch(central_circle)

    # Interior edges
    edge_width = 2.5

    # "/" interior edge (top-right to bottom-left)
    ax.plot(
        [0.5, -0.5],
        [0.5, -0.5],
        color=colors["/"],
        linewidth=edge_width,
        label="/ interior edge",
        alpha=0.8,
    )

    # "\" interior edge (top-left to bottom-right)
    ax.plot(
        [-0.5, 0.5],
        [0.5, -0.5],
        color=colors["\\"],
        linewidth=edge_width,
        label="\\ interior edge",
        alpha=0.8,
    )

    # "V" interior edges - two edges from bottom-middle to top corners
    # Bottom midpoint in the V triangulation
    bm_x, bm_y = 0, -0.5
    # V interior edge 1: bottom-middle to top-left
    ax.plot(
        [bm_x, -0.5],
        [bm_y, 0.5],
        color=colors["V"],
        linewidth=edge_width,
        label="V interior edge 1",
        alpha=0.8,
    )
    # V interior edge 2: bottom-middle to top-right
    ax.plot(
        [bm_x, 0.5],
        [bm_y, 0.5],
        color=colors["V"],
        linewidth=edge_width,
        label="V interior edge 2",
        alpha=0.8,
    )

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel("$r$ = (strike distance) / width", fontsize=12)
    ax.set_ylabel("$s$ = (dip distance) / height", fontsize=12)
    ax.set_title("Triangulation Selection Strategy", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    ticks = np.arange(-0.7, 0.8, 0.1)  # From -0.7 to 0.7 in steps of 0.1
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.grid(True, alpha=0.6, linewidth=1.2)

    # # Center reference lines
    # ax.axhline(0, color="gray", linestyle=":", alpha=0.8, linewidth=edge_width)
    # ax.axvline(0, color="gray", linestyle=":", alpha=0.8, linewidth=edge_width)

    # Rectangle outline for the fault boundary
    rect_outline = Rectangle(
        (-0.5, -0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=edge_width
    )
    ax.add_patch(rect_outline)

    # Professional legend
    legend_elements = [
        patches.Patch(
            facecolor=colors["/"],
            alpha=1,
            edgecolor=colors["/"],
            label="Forward slash (/) regions",
        ),
        patches.Patch(
            facecolor=colors["\\"],
            alpha=1,
            edgecolor=colors["\\"],
            label="Backslash (\\) regions",
        ),
        patches.Patch(
            facecolor=colors["V"],
            alpha=1,
            edgecolor=colors["V"],
            label="V-pattern region",
        ),
        Line2D(
            [0], [0], color=colors["/"], linewidth=2, alpha=0.8, label="/ interior edge"
        ),
        Line2D(
            [0],
            [0],
            color=colors["\\"],
            linewidth=2,
            alpha=0.8,
            label="\\ interior edge",
        ),
        Line2D(
            [0],
            [0],
            color=colors["V"],
            linewidth=2,
            alpha=0.8,
            label="V interior edges",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()

    return fig


def main():
    """Generate and save the triangulation schematic."""
    print("Generating triangulation selection schematic...")

    # Create the figure
    fig = create_triangulation_schematic()

    # Save to a string buffer to ensure a trailing newline
    svg_buffer = io.StringIO()
    fig.savefig(
        svg_buffer,
        format="svg",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        metadata={"Date": None},
    )

    svg_content = svg_buffer.getvalue()

    # Make IDs deterministic for reproducible output
    svg_content = make_svg_deterministic(svg_content)

    # Strip trailing whitespace from each line
    lines = svg_content.splitlines()
    stripped_lines = [line.rstrip() for line in lines]
    svg_content = "\n".join(stripped_lines)

    if not svg_content.endswith("\n"):
        svg_content += "\n"

    # Write to file
    output_file = Path("triangulation_schematic.svg")
    with output_file.open("w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"✅ Schematic saved as {output_file}")
    # Optionally display the figure
    # plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
