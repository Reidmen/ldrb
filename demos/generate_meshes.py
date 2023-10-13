""" Generate meshes for examples """
import argparse
from pathlib import Path
from typing import Tuple
import dolfin as df
import sys

import ldrb
import cardiac_geometries


def rectangle_mesh(L, H, nx, ny, path="./meshes/"):
    """Create rectangle mesh"""
    from dolfin import RectangleMesh, MeshFunction, CompiledSubDomain, XDMFFile
    from dolfin import Point

    mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    left = CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = CompiledSubDomain("near(x[0], L) && on_boundary", L=L)
    bottom = CompiledSubDomain("near(x[1], 0) && on_boundary")
    top = CompiledSubDomain("near(x[1], H) && on_boundary", H=H)

    for i, sd in enumerate((left, right, bottom, top)):
        sd.mark(boundaries, i + 1)

    path = Path(path).joinpath("rect_n{}.xdmf".format(ny))
    with XDMFFile(str(path)) as xf:
        xf.write(mesh)
        xf.write(boundaries)

    print("wrote {}".format(path))

    return mesh, boundaries


def square_mesh(nx, path="./meshes/"):
    """Create unit square mesh"""
    from dolfin import UnitSquareMesh, MeshFunction, CompiledSubDomain
    from dolfin import XDMFFile

    mesh = UnitSquareMesh(nx, nx)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    left = CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = CompiledSubDomain("near(x[0], 1) && on_boundary")
    bottom = CompiledSubDomain("near(x[1], 0) && on_boundary")
    top = CompiledSubDomain("near(x[1], 1) && on_boundary")

    for i, sd in enumerate((left, right, bottom, top)):
        sd.mark(boundaries, i + 1)

    path = Path(path).joinpath("square_n{}.xdmf".format(nx))
    with XDMFFile(str(path)) as xf:
        xf.write(mesh)
        xf.write(boundaries)

    print("wrote {}".format(path))

    return mesh, boundaries


def karman_mesh(ny=10, path="./meshes/"):
    """Turek benchmark geometry and measurement mesh: a circle domain around
    the cylinder.

    Requires gmsh and pygmsh, meshio (<3.3.0), h5py (from PyPi).

    Args:
        ny (int):    approx. number of elements in the y direction
        path (str):  Path to mesh directory
    """
    import pygmsh
    import meshio
    import numpy as np
    from pathlib import Path

    geom = pygmsh.built_in.Geometry()

    cylinder = geom.add_circle((0.2, 0, 0), 0.05, lcar=np.pi / ny / 30)
    rectangle = geom.add_rectangle(
        0.0, 2.2, -0.2, 0.21, 0, lcar=0.41 / ny, holes=[cylinder]
    )

    geom.add_physical(rectangle.surface, 0)
    # inlet
    geom.add_physical(rectangle.lines[3], 1)
    # outlet
    geom.add_physical(rectangle.lines[1], 2)
    geom.add_physical(rectangle.lines[0::2] + cylinder.line_loop.lines, 3)

    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)

    Path(path).mkdir(exist_ok=True, parents=True)

    pth_tmp_msh = Path(path).joinpath("karman_meshio.xdmf")
    pth_tmp_bnd = Path(path).joinpath("karman_meshio_bound.xdmf")
    pth_aux_msh = Path(path).joinpath("karman_meshio.h5")
    pth_aux_bnd = Path(path).joinpath("karman_meshio_bound.h5")

    triangle_cells = mesh.get_cells_type("triangle")
    triangle_data = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    line_cells = mesh.get_cells_type("line")
    line_data = mesh.cell_data_dict["gmsh:physical"]["line"]

    triangle_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", triangle_cells)],
        cell_data={"markers": [triangle_data]},
    )

    line_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("line", line_cells)],
        cell_data={"markers": [line_data]},
    )

    meshio.write(str(pth_tmp_msh), triangle_mesh, file_format="xdmf")
    meshio.write(str(pth_tmp_bnd), line_mesh, file_format="xdmf")

    # Create observation mesh: circle around cylinder
    geom = pygmsh.built_in.Geometry()
    circ_inner = geom.add_circle((0.2, 0, 0), 0.05, 0.025)
    circ = geom.add_circle((0.2, 0, 0), 0.1, 0.025, holes=[circ_inner])
    geom.add_physical(circ.plane_surface, label=0)
    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)

    triangle_cells = mesh.get_cells_type("triangle")
    triangle_data = mesh.cell_data_dict["gmsh:physical"]["triangle"]

    triangle_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", triangle_cells)],
        cell_data={"markers": [triangle_data]},
    )

    path_meas = Path(path).joinpath("karman_n{}_cyl_obs.xdmf".format(ny))
    path_aux = Path(path).joinpath("karman_n{}_cyl_obs.h5".format(ny))

    meshio.write(str(path_meas), triangle_mesh, file_format="xdmf")
    print("wrote {}".format(path_meas))

    # import this after writing XDMF file due to possible h5py version
    # conflicts
    from dolfin import XDMFFile, Mesh, MeshFunction, MeshValueCollection

    if meshio.__version__ == "3.3.0":
        raise Exception(
            "Update meshio! There is a bug in version 3.3.0. See" " issue #599"
        )

    mesh = Mesh()
    with XDMFFile(str(pth_tmp_msh)) as xf:
        xf.read(mesh)

    mvc = MeshValueCollection("size_t", mesh)
    with XDMFFile(str(pth_tmp_bnd)) as infile:
        infile.read(mvc, "markers")

    mf = MeshFunction("size_t", mesh, mvc)

    x = mf.array()
    for i in range(len(x)):
        if x[i] not in mvc.values().values():
            x[i] = 0

    mf.set_values(x)

    path = Path(path).joinpath("karman_n{}.xdmf".format(ny))

    with XDMFFile(str(path)) as xf:
        xf.write(mesh)
        xf.write(mf)

    print("wrote {}".format(path))

    # remove temporary XDMF files
    from pathlib import Path

    Path(pth_tmp_msh).unlink()
    Path(pth_tmp_bnd).unlink()
    Path(path_meas).unlink()


def cube_mesh(L, nx, path="./meshes/"):
    """Generate cube mesh.

    Args:
        L (float):  edge length
        nx (int):   number of elements per edge
        path (str): write path
    """
    from dolfin import BoxMesh, MeshFunction, CompiledSubDomain, XDMFFile
    from dolfin import Point

    mesh = BoxMesh(Point(0, 0, 0), Point(L, L, L), nx, nx, nx)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    left = CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = CompiledSubDomain("near(x[0], L) && on_boundary", L=L)
    bottom = CompiledSubDomain("near(x[1], 0) && on_boundary")
    top = CompiledSubDomain("near(x[1], L) && on_boundary", L=L)
    back = CompiledSubDomain("near(x[2], 0) && on_boundary")
    front = CompiledSubDomain("near(x[2], L) && on_boundary", L=L)

    # iterate x:(0, L), y:(0, L), z:(0, L)
    for i, sd in enumerate((left, right, bottom, top, back, front)):
        sd.mark(boundaries, i + 1)

    path = Path(path).joinpath("cube_L{}_n{}.xdmf".format(L, nx))

    with XDMFFile(str(path)) as xf:
        xf.write(mesh)
        xf.write(boundaries)

    print("wrote {}".format(path))

    return mesh, boundaries


def beam_mesh(L, H, W, ny, path="./meshes/"):
    """Generate beam mesh of [Lan+15]_.


    .. [Lan+15] S. Land et al. (2015). Verification of cardiac mechanics
       software: Benchmark problems and solutions for testing active and
       passive material behaviour. Proceedings of the Royal Society A:
       Mathematical, Physical and Engineering Sciences, 471(2184), 20150641.
       https://doi.org/10.1098/rspa.2015.0641

    Args:
        L (float):  beam length
        H (float):  height
        W (float):  width
        ny (int):   number of elements in y direction
        path (str): write path
    """
    from dolfin import (
        BoxMesh,
        MeshFunction,
        CompiledSubDomain,
        XDMFFile,
        Point,
    )

    nx = int(L / H * ny)
    nz = int(W / H * ny)
    mesh = BoxMesh(Point(0, 0, 0), Point(L, H, W), nx, ny, nz)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    left = CompiledSubDomain("near(x[0], 0) && on_boundary")
    bottom = CompiledSubDomain("near(x[2], 0) && on_boundary")
    right = CompiledSubDomain("near(x[0], L) && on_boundary", L=L)

    left.mark(boundaries, 1)
    bottom.mark(boundaries, 2)
    right.mark(boundaries, 3)
    # # iterate x:(0, L), y:(0, L), z:(0, L)
    # for i, sd in enumerate((left, right, bottom, top, back, front)):
    #     sd.mark(boundaries, i + 1)

    path = Path(path).joinpath("beam_ny{}.xdmf".format(ny))

    with XDMFFile(str(path)) as xf:
        xf.write(mesh)
        xf.write(boundaries)

    print("wrote {}".format(path))
    print("number of points: {}".format(mesh.num_vertices()))

    return mesh, boundaries


def bi_ventricular_fibers_from_mesh(path_to_folder: Path | str):
    """Create fibers from bi-ventricular domain from path"""
    xdmf_path = Path(path_to_folder)
    xdmf_path.mkdir(exist_ok=True, parents=True)

    biv_path = xdmf_path.joinpath("bi_ventricular.xdmf")
    mesh = df.Mesh()
    facet_values = df.MeshValueCollection("size_t", mesh, 2)
    print(biv_path.as_posix())
    with df.XDMFFile(mesh.mpi_comm(), biv_path.as_posix()) as xdmf:
        xdmf.read(mesh)
        xdmf.read(facet_values)

    # facet_func = df.MeshFunction("size_t", mesh, facet_values)

    mesh_prev, facet_func_prev = tag_domain_with_markers_prev_fibers(
        mesh=mesh,
        a_endo_lv=0.069,  # 6.9cm
        b_endo_lv=0.025,  # 2.5cm
        c_endo_lv=0.025,  # 2.5cm
        a_epi_lv=0.08,  # 8.0cm
        b_epi_lv=0.039,  # 3.9cm
        c_epi_lv=0.039,  # 3.9cm
        center_lv=(0.0, 0.0, 0.0),
        a_endo_rv=0.070,  # 6.7cm
        b_endo_rv=0.033,  # 3.0cm
        c_endo_rv=0.054,  # 5.1cm
        a_epi_rv=0.075,  # 7.5cm
        b_epi_rv=0.038,  # 3.8cm
        c_epi_rv=0.059,  # 5.9cm
        center_rv=(0.0, 0.0, 0.019),
        base_x=0.0,
    )

    system = ldrb.dolfin_ldrb(
        mesh=mesh_prev,
        fiber_space="P_2",
        ffun=facet_func_prev,
        markers=dict(base=10, rv=20, lv=30, epi=40),
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
        alpha_endo_rv=-120,
        alpha_epi_rv=60,
    )

    mesh_post, facet_func_post = tag_domain_with_markers_post_fibers(
        mesh=mesh_prev,
        a_endo_lv=0.069,  # 6.9cm
        b_endo_lv=0.025,  # 2.5cm
        c_endo_lv=0.025,  # 2.5cm
        a_epi_lv=0.08,  # 8.0cm
        b_epi_lv=0.039,  # 3.9cm
        c_epi_lv=0.039,  # 3.9cm
        center_lv=(0.0, 0.0, 0.0),
        a_endo_rv=0.070,  # 6.7cm
        b_endo_rv=0.033,  # 3.0cm
        c_endo_rv=0.054,  # 5.1cm
        a_epi_rv=0.075,  # 7.5cm
        b_epi_rv=0.038,  # 3.8cm
        c_epi_rv=0.059,  # 5.9cm
        center_rv=(0.0, 0.0, 0.019),
        base_x=0.0,
    )

    vtkfile = df.File(str(xdmf_path.joinpath("vtk/bi_ventricular.pvd")))
    meshfunc_post = df.MeshFunction("size_t", mesh_post, 3)
    meshfunc_post.set_all(0)

    vtkfile << mesh_post
    vtkfile << facet_func_post
    vtkfile << meshfunc_post

    print("wrote domain in path {}".format(biv_path))
    print("number of points: {}".format(mesh_post.num_vertices()))

    print("saving files to visualize fibers")
    names = ["fiber", "sheet", "sheet_normal"]
    directions = [system.fiber, system.sheet, system.sheet_normal]

    hdf5_path = Path(path_to_folder)
    for name, direction in zip(names, directions):
        hdf5_file = hdf5_path.joinpath(f"fibers/bi_ventricular_{name}.h5")

        with df.HDF5File(mesh.mpi_comm(), str(hdf5_file), "w") as hdf5:
            hdf5.write(direction, "/" + name)

        vtk_fiber = df.File(
            str(hdf5_path.joinpath(f"fibers/bi_ventricular_{name}.pvd"))
        )
        vtk_fiber << direction

        print(f"wrote {name} in path {str(hdf5_file)}")


def bi_ventricular_mesh_with_fibers(char_length: float, path="./meshes/"):
    """Create bi-ventricular domain using the ldrb package
    including fibers (fiber, sheet, sheet-normal) directions.

    Args:
        char_length (float):  characteristic length size
        path (str):     write path
    """
    import ldrb

    geometry = cardiac_geometries.create_biv_ellipsoid(
        outdir=Path(path),
        char_length=char_length,
        a_endo_lv=0.069,  # 6.9cm
        b_endo_lv=0.025,  # 2.5cm
        c_endo_lv=0.025,  # 2.5cm
        a_epi_lv=0.08,  # 8.0cm
        b_epi_lv=0.039,  # 3.9cm
        c_epi_lv=0.039,  # 3.9cm
        center_lv_y=0.0,  # coordinates are 0.0
        a_endo_rv=0.070,  # 6.7cm
        b_endo_rv=0.033,  # 3.0cm
        c_endo_rv=0.054,  # 5.1cm
        a_epi_rv=0.075,  # 7.5cm
        b_epi_rv=0.038,  # 3.8cm
        c_epi_rv=0.059,  # 5.9cm
        center_rv_y=0.0,  # coordinate x is 0.0
        center_rv_z=0.02,
    )

    mesh_prev, facet_func_prev = tag_domain_with_markers_prev_fibers(
        mesh=geometry.mesh,
        a_endo_lv=0.069,  # 6.9cm
        b_endo_lv=0.025,  # 2.5cm
        c_endo_lv=0.025,  # 2.5cm
        a_epi_lv=0.08,  # 8.0cm
        b_epi_lv=0.039,  # 3.9cm
        c_epi_lv=0.039,  # 3.9cm
        center_lv=(0.0, 0.0, 0.0),
        a_endo_rv=0.070,  # 6.7cm
        b_endo_rv=0.033,  # 3.0cm
        c_endo_rv=0.054,  # 5.1cm
        a_epi_rv=0.075,  # 7.5cm
        b_epi_rv=0.038,  # 3.8cm
        c_epi_rv=0.059,  # 5.9cm
        center_rv=(0.0, 0.0, 0.02),
        base_x=0.0,
    )

    xdmf_path = Path(path)
    xdmf_path.mkdir(exist_ok=True, parents=True)

    biv_path_prev = xdmf_path.joinpath("bi_ventricular_prev_tag.xdmf")
    with df.XDMFFile(mesh_prev.mpi_comm(), str(biv_path_prev)) as xdmf_prev:
        xdmf_prev.write(mesh_prev)
        xdmf_prev.write(facet_func_prev)

    # Compute the microstructure
    print("Creating fiber, sheet and sheet_normal")
    system = ldrb.dolfin_ldrb(
        mesh=mesh_prev,
        fiber_space="P_2",
        ffun=facet_func_prev,
        markers=dict(base=10, rv=20, lv=30, epi=40),
        alpha_endo_lv=60,  # Fiber angle on the LV endocardium
        alpha_epi_lv=-60,  # Fiber angle on the LV epicardium
        beta_endo_lv=0,  # Sheet angle on the LV endocardium
        beta_epi_lv=0,  # Sheet angle on the LV epicardium
        alpha_endo_rv=60,  # Fiber angle on the RV endocardium
        alpha_epi_rv=-60,  # Fiber angle on the RV epicardium
    )

    mesh_post, facet_func_post = tag_domain_with_markers_post_fibers(
        mesh=mesh_prev,
        a_endo_lv=0.069,  # 6.9cm
        b_endo_lv=0.025,  # 2.5cm
        c_endo_lv=0.025,  # 2.5cm
        a_epi_lv=0.08,  # 8.0cm
        b_epi_lv=0.039,  # 3.9cm
        c_epi_lv=0.039,  # 3.9cm
        center_lv=(0.0, 0.0, 0.0),
        a_endo_rv=0.070,  # 6.7cm
        b_endo_rv=0.033,  # 3.0cm
        c_endo_rv=0.054,  # 5.1cm
        a_epi_rv=0.075,  # 7.5cm
        b_epi_rv=0.038,  # 3.8cm
        c_epi_rv=0.059,  # 5.9cm
        center_rv=(0.0, 0.0, 0.02),
        base_x=0.0,
    )

    # xdmf_path = Path(path)
    # xdmf_path.mkdir(exist_ok=True, parents=True)

    biv_path = xdmf_path.joinpath("bi_ventricular.xdmf")

    biv_path = xdmf_path.joinpath("bi_ventricular.xdmf")
    with df.XDMFFile(mesh_post.mpi_comm(), str(biv_path)) as xdmf:
        xdmf.write(mesh_post)
        xdmf.write(facet_func_post)

    vtkfile = df.File(str(xdmf_path.joinpath("vtk/bi_ventricular.pvd")))
    vtk_meshfunc = df.MeshFunction("size_t", mesh_post, 3)
    vtk_meshfunc.set_all(0)

    vtkfile << mesh_post
    vtkfile << facet_func_post
    vtkfile << vtk_meshfunc

    print("wrote domain in path {}".format(biv_path))
    print("number of points: {}".format(mesh_post.num_vertices()))

    print("saving files to visualize fibers")
    names = ["fiber", "sheet", "sheet_normal"]
    directions = [system.fiber, system.sheet, system.sheet_normal]

    hdf5_path = Path(path)
    for name, direction in zip(names, directions):
        hdf5_file = hdf5_path.joinpath(f"fibers/bi_ventricular_{name}.h5")

        with df.HDF5File(mesh_prev.mpi_comm(), str(hdf5_file), "w") as hdf5:
            hdf5.write(direction, "/" + name)

        vtk_fiber = df.File(
            str(hdf5_path.joinpath(f"fibers/bi_ventricular_{name}.pvd"))
        )
        vtk_fiber << direction

        print(f"wrote {name} in path {str(hdf5_file)}")


def default_markets() -> dict:
    return dict(base=10, rv=20, lv=30, epi=40)


def tag_domain_with_markers_prev_fibers(
    mesh: df.Mesh = None,
    a_endo_lv=1.5,
    b_endo_lv=0.5,
    c_endo_lv=0.5,
    a_epi_lv=2.0,
    b_epi_lv=1.0,
    c_epi_lv=1.0,
    center_lv=(0.0, 0.0, 0.0),
    a_endo_rv=1.45,
    b_endo_rv=1.25,
    c_endo_rv=0.75,
    a_epi_rv=1.75,
    b_epi_rv=1.5,
    c_epi_rv=1.0,
    center_rv=(0.0, 0.5, 0.0),
    base_x=0.0,
    markers=None,
) -> Tuple[df.Mesh, df.MeshFunction]:

    center_lv = df.Point(*center_lv)
    center_rv = df.Point(*center_rv)

    class EndoLV(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return (x[0] - center_lv.x()) ** 2 / a_endo_lv**2 + (
                x[1] - center_lv.y()
            ) ** 2 / b_endo_lv**2 + (
                x[2] - center_lv.z()
            ) ** 2 / c_endo_lv**2 - 1 < 0.01 + df.DOLFIN_EPS and on_boundary

    class Base(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return x[0] - base_x < df.DOLFIN_EPS and on_boundary

    class EndoRV(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return (
                (x[0] - center_rv.x()) ** 2 / a_endo_rv**2
                + (x[1] - center_rv.y()) ** 2 / b_endo_rv**2
                + (x[2] - center_rv.z()) ** 2 / c_endo_rv**2
                - 1
                < 0.05 + df.DOLFIN_EPS
                and (x[0] - center_lv.x()) ** 2 / a_epi_lv**2
                + (x[1] - center_lv.y()) ** 2 / b_epi_lv**2
                + (x[2] - center_lv.z()) ** 2 / c_epi_lv**2
                - 0.99
                > df.DOLFIN_EPS
            ) and on_boundary

    class Epi(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return (
                (x[0] - center_rv.x()) ** 2 / a_epi_rv**2
                + (x[1] - center_rv.y()) ** 2 / b_epi_rv**2
                + (x[2] - center_rv.z()) ** 2 / c_epi_rv**2
                - 0.98
                > df.DOLFIN_EPS
                and (x[0] - center_lv.x()) ** 2 / a_epi_lv**2
                + (x[1] - center_lv.y()) ** 2 / b_epi_lv**2
                + (x[2] - center_lv.z()) ** 2 / c_epi_lv**2
                - 0.98
                > df.DOLFIN_EPS
                and on_boundary
            )

    class EpiEndoRV(df.SubDomain):
        def inside(self, x, on_boundary):
            return (
                df.near(
                    (x[0] - center_lv.x()) ** 2 / a_epi_lv**2
                    + (x[1] - center_lv.y()) ** 2 / b_epi_lv**2
                    + (x[2] - center_lv.z()) ** 2 / c_epi_lv**2,
                    1,
                    0.01,
                )
                and on_boundary
            )

    if markers is None:
        markers = default_markets()

    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    endolv = EndoLV()
    endolv.mark(ffun, markers["lv"])
    base = Base()
    base.mark(ffun, markers["base"])
    endorv = EndoRV()
    endorv.mark(ffun, markers["rv"])
    epi = Epi()
    epi.mark(ffun, markers["epi"])
    epi_endo_rv = EpiEndoRV()
    epi_endo_rv.mark(ffun, markers["epi"])
    mark_facets(mesh, ffun)

    return mesh, ffun


def tag_domain_with_markers_post_fibers(
    mesh: df.Mesh = None,
    a_endo_lv=1.5,
    b_endo_lv=0.5,
    c_endo_lv=0.5,
    a_epi_lv=2.0,
    b_epi_lv=1.0,
    c_epi_lv=1.0,
    center_lv=(0.0, 0.0, 0.0),
    a_endo_rv=1.45,
    b_endo_rv=1.25,
    c_endo_rv=0.75,
    a_epi_rv=1.75,
    b_epi_rv=1.5,
    c_epi_rv=1.0,
    center_rv=(0.0, 0.5, 0.0),
    base_x=0.0,
    markers=None,
) -> Tuple[df.Mesh, df.MeshFunction]:

    center_lv = df.Point(*center_lv)
    center_rv = df.Point(*center_rv)

    class EndoLV(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return (x[0] - center_lv.x()) ** 2 / a_endo_lv**2 + (
                x[1] - center_lv.y()
            ) ** 2 / b_endo_lv**2 + (
                x[2] - center_lv.z()
            ) ** 2 / c_endo_lv**2 - 1 < 0.01 + df.DOLFIN_EPS and on_boundary

    class Base(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return x[0] - base_x < df.DOLFIN_EPS and on_boundary

    class EndoRV(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return (
                (x[0] - center_rv.x()) ** 2 / a_endo_rv**2
                + (x[1] - center_rv.y()) ** 2 / b_endo_rv**2
                + (x[2] - center_rv.z()) ** 2 / c_endo_rv**2
                - 1
                < 0.05 + df.DOLFIN_EPS
                and (x[0] - center_lv.x()) ** 2 / a_epi_lv**2
                + (x[1] - center_lv.y()) ** 2 / b_epi_lv**2
                + (x[2] - center_lv.z()) ** 2 / c_epi_lv**2
                - 0.99
                > df.DOLFIN_EPS
            ) and on_boundary

    class Epi(df.SubDomain):
        def inside(self, x, on_boundary) -> None:
            return (
                (x[0] - center_rv.x()) ** 2 / a_epi_rv**2
                + (x[1] - center_rv.y()) ** 2 / b_epi_rv**2
                + (x[2] - center_rv.z()) ** 2 / c_epi_rv**2
                - 0.98
                > df.DOLFIN_EPS
                and (x[0] - center_lv.x()) ** 2 / a_epi_lv**2
                + (x[1] - center_lv.y()) ** 2 / b_epi_lv**2
                + (x[2] - center_lv.z()) ** 2 / c_epi_lv**2
                - 0.98
                > df.DOLFIN_EPS
                and on_boundary
            )

    if markers is None:
        markers = default_markets()

    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    endolv = EndoLV()
    endolv.mark(ffun, markers["lv"])
    base = Base()
    base.mark(ffun, markers["base"])
    endorv = EndoRV()
    endorv.mark(ffun, markers["rv"])
    epi = Epi()
    epi.mark(ffun, markers["epi"])

    mark_facets(mesh, ffun)

    return mesh, ffun


def mark_facets(mesh: df.Mesh, ffun: df.MeshFunction) -> None:
    for facet in df.facets(mesh):
        if ffun[facet] == 2**64 - 1:
            ffun[facet] = 0

        mesh.domains().set_marker((facet.index(), ffun[facet]), 2)


def get_parser():
    """Get arguments parser.

    Returns:
        parser
    """
    parser = argparse.ArgumentParser(
        description="Generate meshes for examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--navierstokes",
        action="store_true",
        help="Generate meshes for NavierStokes examples",
    )
    parser.add_argument(
        "--hyperelasticity",
        action="store_true",
        help="Generate meshes for Hyperelasticity examples",
    )
    parser.add_argument(
        "--rectangle",
        metavar="NY",
        type=int,
        default=-1,
        help="Rectangle mesh with dim 5 x 1, NY elements in Y",
    )
    parser.add_argument(
        "--karman",
        metavar="N",
        type=int,
        default=-1,
        help="Karman vortex street mesh with NY elements in Y",
    )
    parser.add_argument(
        "--square",
        metavar="N",
        type=int,
        default=-1,
        help="Unit square mesh with N x N elements",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./meshes/",
        help="Path where meshes are written",
    )
    parser.add_argument(
        "--cube",
        metavar="N",
        type=int,
        default=-1,
        help="Cube mesh with edge L=1e-3 and N x N elements",
    )
    parser.add_argument(
        "--beam",
        metavar="Ny",
        type=int,
        default=-1,
        help="Beam mesh (default 10 x 1 x 1) and 10Ny x Ny x " "Ny elements",
    )
    parser.add_argument(
        "--biventricular_with_fibers",
        metavar="char_length",
        type=float,
        default=-1,
        help="Bi-ventricular mesh (default) with discretization "
        "characteristic length char_length and defaul fiber configuration",
    )
    parser.add_argument(
        "--biventricular_fibers_from_mesh",
        metavar="path_to_folder",
        type=str,
        default=0,
        help="Creates fibers from a biventricular domain "
        "with path specified by the user",
    )
    return parser


def default_meshes_navierstokes():
    """Generate the default meshes for the NavierStokes examples."""
    print("Generate NavierStokes default meshes")
    square_mesh(32)
    square_mesh(16)
    karman_mesh(ny=20)


def default_meshes_hyperelasticity():
    """Generate the default meshes for the Hyperelasticity examples."""
    print("Generate Hyperelasticity default meshes")
    # cube_mesh(1e-3, 16)
    beam_mesh(1e-2, 1e-3, 1e-3, 2)


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)

    if len(sys.argv) > 1:

        if args.navierstokes:
            default_meshes_navierstokes()
        if args.hyperelasticity:
            default_meshes_hyperelasticity()

        if not args.navierstokes:
            if args.karman > 0:
                karman_mesh(ny=args.karman, path=args.path)
            if args.square > 0:
                square_mesh(args.square, path=args.path)
            if args.rectangle > 0:
                rectangle_mesh(
                    5, 1, 5 * args.rectangle, args.rectangle, path=args.path
                )

        if not args.hyperelasticity:
            if args.cube > 0:
                cube_mesh(1e-3, args.cube, path=args.path)
            if args.beam > 0:
                beam_mesh(1e-2, 1e-3, 1e-3, args.beam, path=args.path)
            if args.biventricular_with_fibers > 0:
                bi_ventricular_mesh_with_fibers(
                    args.biventricular_with_fibers, path=args.path
                )
            if isinstance(args.biventricular_fibers_from_mesh, str):
                bi_ventricular_fibers_from_mesh(
                    args.biventricular_fibers_from_mesh
                )

    else:

        print("Generate default meshes")
        default_meshes_navierstokes()
        default_meshes_hyperelasticity()
