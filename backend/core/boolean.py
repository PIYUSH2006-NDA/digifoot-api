import trimesh

def generate_final_insole(base_insole, foot_mesh):
    return trimesh.boolean.difference(
        [base_insole, foot_mesh],
    )