
def number_of_default_boxes_per_cell(aspect_ratios: list) -> list:

    num_defaults_per_cell = []

    for aspect_ratio in aspect_ratios:

        num_defaults = 2 + len(aspect_ratio) * 2
        num_defaults_per_cell.append(num_defaults)

    return num_defaults_per_cell

def total_number_of_default_boxes(num_defaults_per_cell, feature_map_sizes):

    total = 0

    for ratio, size in zip(num_defaults_per_cell, feature_map_sizes):

        total += ratio * size * size

    return total

