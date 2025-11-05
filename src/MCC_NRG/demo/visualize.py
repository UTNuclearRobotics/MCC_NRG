import open3d as o3d
import numpy as np

def load_obj_mesh_or_points(path):
    # Try as triangle mesh first
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    if mesh is not None and not mesh.is_empty() and mesh.has_triangles():
        mesh.compute_vertex_normals()
        return mesh, None  # (mesh, pcd)

    # Fallback: parse vertices (“v x y z [r g b]”) and build a point cloud
    pts, cols = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("v "):  # vertex line
                continue
            parts = line.strip().split()
            vals = list(map(float, parts[1:]))
            if len(vals) < 3:
                continue
            x, y, z = vals[:3]
            pts.append([x, y, z])
            if len(vals) >= 6:
                r, g, b = vals[3:6]
                # handle either 0–1 or 0–255
                if max(r, g, b) > 1.0:
                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                cols.append([r, g, b])

    if not pts:
        raise RuntimeError("OBJ contains no faces and no vertices (unsupported).")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    if cols:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
    return None, pcd  # (mesh, pcd)


if __name__ == "__main__":
    mesh, pcd = load_obj_mesh_or_points("spyro.obj")
    geom = mesh if mesh is not None else pcd
    o3d.visualization.draw_geometries([geom], window_name="OBJ viewer")
