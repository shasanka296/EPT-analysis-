import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


trajectory_file = "merged.xtc"
topology_file = "new_plom.gro"
u = mda.Universe(topology_file, trajectory_file, in_memory=False)
print("loaded in memory")




molecules = u.select_atoms("resname EPC").residues

vector_apinr = {}
vectro_cross = {}

time_list = []
angle_average_list_time = []
extended_cryst=[]
cryst=[]
ori_average_list_time = []
dist_average_list_time = []
standard_devations=[]


for ts in u.trajectory[::1000]:

    time = ts.time
    print("Time {}".format(time))

    angle_average_list = []

    oreintation_angle_avg = []
    distances = []


    def crystalanity():
        AA = np.mean((np.array(angle_average_list)))
        OA = np.mean((np.array(oreintation_angle_avg)))
        DA = np.average(np.array(distances))
        return AA / (DA * OA)


    def extended_crystalinty():
        ideal_extened_cryst = 0.0010237515512878134
        AAsig = np.std(angle_average_list, ddof=1)
        OAsig = np.std(distances, ddof=1)
        DAsig = np.std(oreintation_angle_avg, ddof=1)
        standard_devations.append([AAsig, OAsig, DAsig])
        cystalnity = crystalanity()
        return (cystalnity / (AAsig * OAsig * DAsig * ideal_extened_cryst))


    atoms_in_box = u.select_atoms(f"prop abs z >= {63} and resname EPC").residues

    box_center = 0.5 * (u.dimensions[:3].min(axis=0) + u.dimensions[:3].max(axis=0))

    for molecule in atoms_in_box:
        angles = []
        oreintation_angle = []

        molecule_center_of_mass = molecule.atoms.center_of_mass()

        vector_molecule_com_to_box_center = molecule_center_of_mass - box_center

        group_selection = "name C13"
        N = "name N1"
        sulfur = "name S1 "
        carbon = "name C1"
        carbon2 = "name C12"
        atom = molecule.atoms
        group_atoms = atom.select_atoms(group_selection)
        group_center_of_mass = group_atoms.positions[0]
        v_vector = atom.select_atoms(N).positions[0]
        vector_group_to_box_center = group_center_of_mass - box_center
        r_vector = group_center_of_mass - v_vector
        neighbors = u.select_atoms(f"(around   3 resid {molecule.resid}) and resname EPC").residues

        v1_vector = atom.select_atoms(N).positions[0]
        sulfur_vector = atom.select_atoms(sulfur).positions[0]
        carbon_vector = atom.select_atoms(carbon).positions[0]
        carbon_vector2 = atom.select_atoms(carbon2).positions[0]

        in_plane_vector_for_cross = carbon_vector - sulfur_vector
        in2 = carbon_vector2 - sulfur_vector
        perpendicular_vector_to_the_plane_of_the_molecule = np.cross(in_plane_vector_for_cross, r_vector)
        perp_2 = np.cross(in2, r_vector)

        for neighbor in neighbors.residues:
            neighbor_center_of_mass = neighbor.atoms.center_of_mass()

            sulfur_vector_ne = neighbor.atoms.select_atoms(sulfur).positions[0]
            carbon_vector_ne = neighbor.atoms.select_atoms(carbon).positions[0]

            in_plane_vector_for_cross_ne = carbon_vector_ne - sulfur_vector_ne

            n_atom = neighbor.atoms
            neighbor_n_atoms = n_atom.select_atoms(N).positions[0]
            neighbor_group_atoms = n_atom.select_atoms(group_selection)
            neighbor_group_center_of_mass = neighbor_group_atoms.positions[0]
            vector_neighbor_group_to_box_center = neighbor_group_center_of_mass - box_center
            r_vector_neighbor = neighbor_group_center_of_mass - neighbor_n_atoms
            perpendicular_vector_to_the_plane_of_the_molecule_ne = np.cross(in_plane_vector_for_cross_ne,
                                                                            r_vector_neighbor)

            orientation = min(np.arccos(np.dot(perpendicular_vector_to_the_plane_of_the_molecule_ne,
                                               perpendicular_vector_to_the_plane_of_the_molecule) /
                                        (np.linalg.norm(
                                            perpendicular_vector_to_the_plane_of_the_molecule_ne) * np.linalg.norm(
                                            perpendicular_vector_to_the_plane_of_the_molecule))),
                              np.arccos(np.dot(perpendicular_vector_to_the_plane_of_the_molecule_ne,
                                               perp_2) /
                                        (np.linalg.norm(
                                            perpendicular_vector_to_the_plane_of_the_molecule_ne) * np.linalg.norm(
                                            perp_2)))
                              )

            angle = np.arccos(np.dot(r_vector, r_vector_neighbor) /
                              (np.linalg.norm(r_vector) * np.linalg.norm(r_vector_neighbor)))
            distance = np.linalg.norm(molecule_center_of_mass - neighbor_center_of_mass)

            angles.append(np.degrees(angle))
            oreintation_angle.append(np.degrees(orientation))
            distances.append(distance)


            vector_apinr[f"{r_vector}"] = [r_vector_neighbor, np.degrees(angle)]
            vectro_cross[f"{perpendicular_vector_to_the_plane_of_the_molecule} or {perp_2}"] = [perpendicular_vector_to_the_plane_of_the_molecule_ne, np.degrees(orientation)]
        try:
            angle_average_list.append(max(angles))

        except ValueError:
            continue

        try:
            oreintation_angle_avg.append(min(oreintation_angle))

        except ValueError:
            continue



    angle_average_list_time.append(np.mean((np.array(angle_average_list))))
    ori_average_list_time.append(np.mean((np.array(oreintation_angle_avg))))
    dist_average_list_time.append(np.average(np.array(distances)))
    extended_cryst.append(extended_crystalinty())
    cryst.append(crystalanity())
    time_list.append(time)


time_array = np.array(time_list)
angle_average_array = np.array(angle_average_list)
ori_array=np.array(ori_average_list_time)
dist_array=np.array(dist_average_list_time)
exten_array=np.array(extended_cryst)
cryst_array=np.array(cryst)

plt.figure()
plt.plot(time_array, cryst_array)
plt.xlabel("Time (ps)")
plt.ylabel("cryst (1/A)")
plt.title("cryst vs Time")
plt.savefig("cryst")

plt.figure()
plt.plot(time_array, exten_array)
plt.xlabel("Time (ps)")
plt.ylabel("cryst ")
plt.title("cryst vs Time")
plt.savefig("Ex_cryst")

df = pd.DataFrame({'time': time_array, 'ext': exten_array})

df.to_csv('data_ext.csv', index=False)

df = pd.DataFrame({'time': time_array, 'crst': cryst_array})

df.to_csv('data_crst.csv', index=False)

data = list(vector_apinr.items())

df = pd.DataFrame(data, columns=['Key', 'Value'])

df.to_csv("angle_vectors.csv", index=False, lineterminator=" done \n \n \n")

data = list(vectro_cross.items())

df = pd.DataFrame(data, columns=['Key', 'Value'])

df.to_csv("ori_vectors.csv", index=False, lineterminator=" done \n \n \n")


df = pd.DataFrame( {'Time': time_array, 'distance': dist_array})

df.to_csv("distance.csv", index=False, lineterminator=" done \n \n \n")


df = pd.DataFrame( {'Time': time_array, 'standard_dev(A,O,D)': standard_devations})

df.to_csv("std.csv", index=False, lineterminator=" done \n \n \n")
