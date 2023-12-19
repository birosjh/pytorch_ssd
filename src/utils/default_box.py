def total_number_of_default_boxes(num_defaults_per_cell, feature_map_sizes):
    total = 0

    for ratio, size in zip(num_defaults_per_cell, feature_map_sizes):
        total += ratio * size * size

    return total
