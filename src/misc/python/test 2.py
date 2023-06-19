def normalize(points_3d):
    min_x = min(point[0] for point in points_3d)
    max_x = max(point[0] for point in points_3d)
    min_y = min(point[1] for point in points_3d)
    max_y = max(point[1] for point in points_3d)
    min_z = min(point[2] for point in points_3d)
    max_z = max(point[2] for point in points_3d)

    normalized_points = []
    for point in points_3d:
        x, y, z = point
        x_norm = (x - min_x) / (max_x - min_x)
        y_norm = (y - min_y) / (max_y - min_y)
        z_norm = (z - min_z) / (max_z - min_z)
        normalized_points.append((x_norm, y_norm, z_norm))

    return normalized_points


def orthographic_projection(points_3d):
    points_2d = []
    for point in points_3d:
        x, y, _ = point  # Ignore the normalized z-coordinate
        # Apply orthographic projection
        x_2d = x
        y_2d = y
        points_2d.append((x_2d, y_2d))
    return points_2d


# Example usage
points_3d = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
normalized_points = normalize(points_3d)
points_2d = orthographic_projection(normalized_points)
print(points_2d)

