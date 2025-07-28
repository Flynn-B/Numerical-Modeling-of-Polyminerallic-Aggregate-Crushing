"""
Simple python script to transform STL files of clumps into ASCII VTK files of format UNSTRUCTURED_GRID.
Format matches Sphere Example in LIGGGHTS-with-bonds; https://github.com/richti83/LIGGGHTS-WITH-BONDS/tree/master/examples/bondspackage/sphere
Dependencies: pip install gmsh meshio vtk
- FB 
"""

import gmsh
import meshio
import os

import vtk

# Function to manually write legacy VTK UnstructuredGrid in Version 2.0 format
def write_legacy_vtk(unstructured_grid, filename):
    with open(filename, 'w') as f:
        # Write header
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Unstructured Grid\n") # Comment line
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Write points
        points = unstructured_grid.GetPoints()
        num_points = points.GetNumberOfPoints()
        f.write(f"POINTS {num_points} double\n")
        for i in range(num_points):
            x, y, z = points.GetPoint(i)
            f.write(f"{x} {y} {z}\n")
        
        # Write cells
        f.write("\n")
        num_cells = unstructured_grid.GetNumberOfCells()
        total_size = 0
        cell_list = []
        
        for i in range(num_cells):
            cell = unstructured_grid.GetCell(i)
            pt_ids = cell.GetPointIds()
            ids = [pt_ids.GetId(j) for j in range(pt_ids.GetNumberOfIds())]
            cell_list.append(ids)
            total_size += len(ids) + 1
        
        f.write(f"CELLS {num_cells} {total_size}\n")
        for ids in cell_list:
            f.write(f"{len(ids)} {' '.join(map(str, ids))}\n")
        
        # Write cell types
        f.write("\n")
        f.write(f"CELL_TYPES {num_cells}\n")
        for i in range(num_cells):
            f.write(f"{unstructured_grid.GetCellType(i)}\n")

    print(f"Wrote legacy VTK file: {filename}")

def stl_to_3d_mesh_save(stl_file: str, output_file: str):
    assert stl_file.endswith(".stl")
    assert output_file.endswith(".vtk") or output_file.endswith(".msh")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(stl_file)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10.0)

    surfaces = gmsh.model.getEntities(2)
    surface_tags = [s[1] for s in surfaces]

    gmsh.model.geo.synchronize()
    sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)

    temp_msh = "temp_output.msh"
    gmsh.write(temp_msh)
    gmsh.finalize()

    if output_file.endswith(".vtk"):
        mesh = meshio.read(temp_msh)
        cells = [c for c in mesh.cells if c.type == "tetra"]
        if not cells:
            raise RuntimeError("No tetrahedral cells found.")
        meshio.write(output_file, meshio.Mesh(mesh.points, cells), file_format="vtk", binary=False)
        os.remove(temp_msh)

        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(output_file)
        reader.Update()
        grid = reader.GetOutput()

        os.remove(output_file)
        write_legacy_vtk(grid, f"results/{output_file}")

    elif output_file.endswith(".msh"):
        os.rename(temp_msh, output_file)

    print(f"Saved and done")

if __name__ == "__main__":
    #stl_to_3d_mesh_save("SphereLeft.stl", "SphereLeft.vtk")
    file_names = os.listdir("STLOutput")
    file_names.remove("TriangleFeature_0.stl") # Cube boundary box
    file_names.remove("TriangleFeature_-1.stl") # Inverse
    for file in file_names:
        assert file.endswith('.stl')
        stl_to_3d_mesh_save(f"STLOutput/{file}",f"{file[:-4]}.vtk")