from dolfinx import fem, io, mesh
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

ncells = 100
domain = mesh.create_unit_square(comm, ncells, ncells)

V = fem.FunctionSpace(domain, ("CG", 1))

# create two arbitrary function at three times
phis = []
xis = []
t = [0.0, 0.5, 1.0]
for i in t:
    phi = fem.Function(V)
    phi.name = "phi"  # This determines the name in reader.set_active_scalars("phi")
    xi = fem.Function(V)
    xi.name = "xi"
    phi.interpolate(lambda x: (i + 0.1) * x[0])
    xi.interpolate(lambda x: (i + 1.1) * x[0])
    phis.append(phi)
    xis.append(xi)

# option 1 save
# Create xdmf file for saving time series
xdmf = io.XDMFFile(comm, "phi.xdmf", "w")
xdmf.write_mesh(domain)
for i in range(len(t)):
    xdmf.write_function(phis[i], t[i])
    xdmf.write_function(xis[i], t[i])
xdmf.close()

# option 2 save
with io.XDMFFile(comm, "phi01.xdmf", "w") as f:
    f.write_mesh(domain)

    for i in range(len(t)):
        f.write_function(phis[i], t[i])
        f.write_function(xis[i], t[i])

# option 3 save
with io.XDMFFile(comm, "phi02.xdmf", "w") as f:
    f.write_mesh(domain)

with io.XDMFFile(comm, "phi02.xdmf", "a") as f:
    for i in range(len(t)):
        f.write_function(phis[i], t[i])
        f.write_function(xis[i], t[i])
